"""
Scheduler of the context-summarizer experiment. Creates the Tasks in the EASE core, starts them, checks that they
finish, archives the results and informs selected people by e-mail. It is run periodically (e.g. every 10 minutes by
cron through tools/experiment_scheduler.sh, which also passes the LLM tokens stored in frontEASE).

Work is added in batches (e.g. one generator family at a time, because of the API budget):
    python tools/experiment_scheduler.py enqueue --batch G1-phase1 --generators G1

Every `tick`:
  1. checks the Tasks started by the scheduler (core REST API):
     - FINISH with all valid iterations -> done; FINISH with fewer (consecutive invalid / max iterations) and BREAK
       caused by a summary failure -> attrition (recorded, not repeated, as in the paper),
     - other BREAK (LLM API error, ...) or a lost Task -> infrastructure failure: the item is repeated as a new Task at
       most `max_retries` times; after that the scheduler halts (no new Tasks) and sends an alert,
     - a running Task without progress for `stuck_hours` -> alert;
  2. archives finished Tasks: zip of out_task/t_<id> into `archive_dir`, then the Task is archived in the core (the
     folder is renamed to a_<id>, frontEASE removes the Task at its next synchronization);
  3. starts queued items while fewer than `max_concurrent` Tasks run in the core (all Tasks, not only own ones);
  4. sends pending alerts (at most one e-mail per `alert_interval_minutes`, a halt immediately) and the daily report
     at `report_time`.

Files (experiment/ next to out_task, not versioned):
    experiment/scheduler.json   configuration (see tools/experiment_scheduler.example.json)
    experiment/state.sqlite     state of all items, attempts and alerts
    experiment/archive/         zipped Task folders (moved to the NAS by experiment_scheduler.sh)
    experiment/scheduler.log    log

Commands: enqueue, dequeue, tick, status, pause, resume, requeue, report, test-email, config-get.
"""

import argparse
import contextlib
import csv
import datetime as dt
import glob
import json
import logging
import os
import shutil
import smtplib
import sqlite3
import sys
import types
from email.message import EmailMessage

import requests

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, CORE_DIR)
import create_smoke_tasks  # noqa: E402

EXP_DIR = os.path.join(CORE_DIR, "experiment")
OUT_TASK = os.path.join(CORE_DIR, "out_task")
PLAN_FILE = os.path.join(TOOLS_DIR, "experiment_plan.json")

# Task states of the core (fopimt/task_dto.py)
CREATED, INIT, RUN, PAUSED, STOP, FINISH, BREAK = range(7)
STATE_NAMES = {0: "CREATED", 1: "INIT", 2: "RUN", 3: "PAUSED", 4: "STOP", 5: "FINISH", 6: "BREAK"}

# Item statuses
QUEUED, RUNNING, DONE, ATTRITION, FAILED, STOPPED = "queued", "running", "done", "attrition", "failed", "stopped"
FINAL = (DONE, ATTRITION, FAILED, STOPPED)

DEFAULT_CONFIG = {
    "core_url": "http://localhost:8086",
    "author": "",  # e-mail of the frontEASE user that owns the Tasks and whose stored LLM tokens are used
    "max_concurrent": 4,
    "max_retries": 2,
    "stuck_hours": 4,
    "report_time": "08:00",
    "timezone": "Europe/Prague",
    "alert_interval_minutes": 60,
    "archive_dir": "experiment/archive",  # relative to the core directory
    "archive_target": "",  # host directory (e.g. NAS mount) the zips are moved to by experiment_scheduler.sh
    "smtp": {
        "host": "",
        "port": 587,
        "security": "starttls",  # starttls | ssl | none
        "username": "",
        "password": "",
        "from": "",
    },
    "recipients": [],
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    key TEXT PRIMARY KEY,           -- G1-F1-C6-S2-r03
    seq INTEGER NOT NULL,           -- start order
    batch TEXT NOT NULL,
    generator TEXT NOT NULL, function TEXT NOT NULL, cell TEXT NOT NULL, scenario TEXT NOT NULL,
    rep INTEGER NOT NULL,
    status TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 0,  -- number of started attempts
    task_id TEXT,
    started_at TEXT, finished_at TEXT,
    last_iteration INTEGER, last_progress_at TEXT, stuck_alerted INTEGER NOT NULL DEFAULT 0,
    reason TEXT,
    result TEXT                     -- JSON summary of the last attempt
);
CREATE TABLE IF NOT EXISTS attempts (
    task_id TEXT PRIMARY KEY,
    key TEXT NOT NULL, attempt INTEGER NOT NULL,
    started_at TEXT, finished_at TEXT,
    status TEXT, reason TEXT, result TEXT,
    archive TEXT,                   -- zip file name; NULL until archived
    archived_in_core INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL, level TEXT NOT NULL, text TEXT NOT NULL, sent_at TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


####################################################################
#########  Helpers
####################################################################
def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def iso(t: dt.datetime | None) -> str | None:
    return t.isoformat() if t else None


def parse_iso(s: str | None) -> dt.datetime | None:
    return dt.datetime.fromisoformat(s) if s else None


def hours(delta: dt.timedelta) -> str:
    return f"{delta.total_seconds() / 3600:.1f} h"


def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_config(exp_dir: str) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    user = load_json(os.path.join(exp_dir, "scheduler.json"), {})
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def item_key(g, f, c, s, rep) -> str:
    return f"{g}-{f}-{c}-{s}-r{rep:02d}"


def parse_reps(text: str, maximum: int) -> list[int]:
    reps = set()
    for part in text.split(","):
        a, _, b = part.partition("-")
        reps.update(range(int(a), int(b or a) + 1))
    if not reps or min(reps) < 1 or max(reps) > maximum:
        raise ValueError(f"Repetitions must be within 1-{maximum}.")
    return sorted(reps)


@contextlib.contextmanager
def tick_lock(exp_dir: str):
    """Only one tick at a time (a slow tick must not overlap with the next cron call)."""
    path = os.path.join(exp_dir, "tick.lock")
    f = open(path, "w")
    try:
        try:
            import fcntl

            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:  # Windows (tests)
            pass
        except OSError:
            raise RuntimeError("Another tick is running.")
        yield
    finally:
        f.close()


####################################################################
#########  Core REST API
####################################################################
class CoreClient:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def states(self, task_ids: list[str] | None = None) -> dict[str, dict]:
        """Info of the given Tasks, or of all Tasks of the core (task_ids None)."""
        params = [("task_ids", t) for t in task_ids] if task_ids else None
        r = requests.get(f"{self.url}/task/status", params=params, timeout=60)
        r.raise_for_status()
        return {t["id"]: t for t in r.json()}

    def create(self, config: dict) -> str:
        r = requests.post(f"{self.url}/task", timeout=60)
        r.raise_for_status()
        task_id = r.json()["id"]
        r = requests.put(f"{self.url}/task/{task_id}", json=config, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"Task {task_id} could not be initialized: {r.status_code} {r.text[:500]}")
        return task_id

    def run(self, task_id: str) -> dict:
        r = requests.patch(f"{self.url}/task/{task_id}/run", timeout=60)
        r.raise_for_status()
        return r.json()

    def archive(self, task_id: str) -> bool:
        r = requests.delete(f"{self.url}/task/{task_id}", timeout=60)
        r.raise_for_status()
        return bool(r.json())


def state_of(info: dict) -> int:
    s = info.get("state")
    if isinstance(s, str):
        return {v: k for k, v in STATE_NAMES.items()}.get(s, -1)
    return int(s) if s is not None else -1


####################################################################
#########  Mail
####################################################################
def send_mail(cfg: dict, subject: str, body: str) -> bool:
    """Returns False if the mail is not configured (the text is only logged)."""
    smtp, recipients = cfg["smtp"], cfg["recipients"]
    if not smtp.get("host") or not recipients:
        logging.info("Mail not configured, not sent: %s\n%s", subject, body)
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp.get("from") or smtp.get("username")
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)
    security = smtp.get("security", "starttls")
    cls = smtplib.SMTP_SSL if security == "ssl" else smtplib.SMTP
    with cls(smtp["host"], int(smtp.get("port", 587)), timeout=60) as s:
        if security == "starttls":
            s.starttls()
        if smtp.get("username"):
            s.login(smtp["username"], smtp.get("password", ""))
        s.send_message(msg)
    return True


####################################################################
#########  Scheduler
####################################################################
class Scheduler:
    def __init__(self, exp_dir: str = EXP_DIR, out_task: str = OUT_TASK, plan_file: str = PLAN_FILE,
                 core: CoreClient | None = None, now=utcnow, mailer=send_mail):
        self.exp_dir = exp_dir
        self.out_task = out_task
        os.makedirs(exp_dir, exist_ok=True)
        self.cfg = load_config(exp_dir)
        self.plan = load_json(plan_file)
        self.core = core or CoreClient(self.cfg["core_url"])
        self.now = now
        self.mailer = mailer
        self.archive_dir = os.path.join(CORE_DIR, self.cfg["archive_dir"]) if not os.path.isabs(
            self.cfg["archive_dir"]) else self.cfg["archive_dir"]
        self.db = sqlite3.connect(os.path.join(exp_dir, "state.sqlite"))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.tokens = None  # frontEASE tokens (tick --tokens-stdin)

    # ---------------------------------------------------------------- meta
    def meta(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value):
        if value is None:
            self.db.execute("DELETE FROM meta WHERE key = ?", (key,))
        else:
            self.db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(value)))
        self.db.commit()

    def alert(self, text: str, level: str = "alert"):
        logging.warning("ALERT [%s] %s", level, text)
        self.db.execute("INSERT INTO alerts (created_at, level, text) VALUES (?, ?, ?)",
                        (iso(self.now()), level, text))
        self.db.commit()

    def halt(self, reason: str):
        self.set_meta("halted", reason)
        self.alert(f"Scheduler HALTED, no new Tasks are started: {reason}\n"
                   "Running Tasks continue. Resolve the problem, then run `experiment_scheduler.py resume`.",
                   level="halt")

    # ---------------------------------------------------------------- enqueue
    def enqueue(self, batch: str, generators, functions=None, cells=None, scenarios=None, reps=None) -> int:
        p = self.plan
        functions = functions or list(p["functions"])
        cells = cells or list(p["cells"])
        scenarios = scenarios or list(p["scenarios"])
        reps = reps or list(range(1, p["repetitions"] + 1))
        for name, values, catalog in (("generator", generators, p["generators"]), ("function", functions,
                                      p["functions"]), ("cell", cells, p["cells"]), ("scenario", scenarios,
                                      p["scenarios"])):
            unknown = [v for v in values if v not in catalog]
            if unknown:
                raise ValueError(f"Unknown {name}: {unknown} (known: {list(catalog)})")
        seq = self.db.execute("SELECT COALESCE(MAX(seq), 0) FROM items").fetchone()[0]
        added = 0
        # order: repetition first, so that an interrupted batch has the same number of repetitions everywhere
        for rep in reps:
            for g in generators:
                for f in functions:
                    for c in cells:
                        for s in scenarios:
                            _, summary = create_smoke_tasks.CELLS[p["cells"][c]]
                            if summary is None and s != "S1":
                                continue  # cells without a summary do not depend on the summarizer (run once, S1)
                            key = item_key(g, f, c, s, rep)
                            if self.db.execute("SELECT 1 FROM items WHERE key = ?", (key,)).fetchone():
                                continue
                            seq += 1
                            self.db.execute(
                                "INSERT INTO items (key, seq, batch, generator, function, cell, scenario, rep, status)"
                                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (key, seq, batch, g, f, c, s, rep, QUEUED))
                            added += 1
        self.db.commit()
        return added

    def dequeue(self, batch: str | None, keys: list[str] | None) -> int:
        if keys:
            q = f"DELETE FROM items WHERE status = ? AND key IN ({','.join('?' * len(keys))})"
            n = self.db.execute(q, (QUEUED, *keys)).rowcount
        else:
            n = self.db.execute("DELETE FROM items WHERE status = ? AND batch = ?", (QUEUED, batch)).rowcount
        self.db.commit()
        return n

    def requeue(self, keys: list[str]) -> int:
        n = 0
        for key in keys:
            n += self.db.execute(
                "UPDATE items SET status = ?, task_id = NULL, reason = NULL, stuck_alerted = 0 WHERE key = ? AND"
                f" status IN ({','.join('?' * len(FINAL))})", (QUEUED, key, *FINAL)).rowcount
        self.db.commit()
        return n

    # ---------------------------------------------------------------- tick
    def tick(self):
        with tick_lock(self.exp_dir):
            try:
                self.poll()
                self.archive_finished()
                if self.meta("halted") is None and self.meta("paused") is None:
                    self.fill()
            except Exception as e:  # unexpected error: report it, keep the scheduler alive for the next tick
                logging.exception("Tick failed")
                self.alert(f"Tick failed: {e!r}")
            for step in (self.send_alerts, self.daily_report):
                try:
                    step()
                except Exception:  # e.g. SMTP unavailable: logged, tried again on the next tick
                    logging.exception("%s failed", step.__name__)

    def poll(self):
        running = self.db.execute("SELECT * FROM items WHERE status = ?", (RUNNING,)).fetchall()
        if not running:
            return
        infos = self.core.states([r["task_id"] for r in running])
        now = self.now()
        for item in running:
            info = infos.get(item["task_id"])
            if info is None:
                self.failure(item, "Task not found in the core (core restarted or Task removed).")
                continue
            state = state_of(info)
            if state == RUN:
                it = info.get("current_iteration")
                if it != item["last_iteration"]:
                    self.db.execute("UPDATE items SET last_iteration = ?, last_progress_at = ? WHERE key = ?",
                                    (it, iso(now), item["key"]))
                elif (not item["stuck_alerted"] and item["last_progress_at"]
                      and now - parse_iso(item["last_progress_at"]) > dt.timedelta(hours=self.cfg["stuck_hours"])):
                    self.alert(f"{item['key']} (Task {item['task_id']}): no progress for "
                               f"{hours(now - parse_iso(item['last_progress_at']))} (iteration {it}).")
                    self.db.execute("UPDATE items SET stuck_alerted = 1 WHERE key = ?", (item["key"],))
            elif state == FINISH:
                valid = info.get("iterations_valid") or 0
                if valid >= self.plan["iterations"]:
                    self.finish(item, DONE, None, info)
                else:
                    self.finish(item, ATTRITION, f"stopped after {valid} valid algorithms "
                                                 "(consecutive invalid / maximum iterations)", info)
            elif state == BREAK:
                log = " | ".join(info.get("log") or [])
                if "Summary failure" in log or "SummaryFailure" in log:
                    self.finish(item, ATTRITION, f"summary failure: {log[-500:]}", info)
                else:
                    self.failure(item, f"Task broke: {log[-500:] or 'no log'}", info)
            elif state in (STOP, PAUSED):
                if state == STOP:
                    self.finish(item, STOPPED, "Task was stopped outside the scheduler.", info)
                    self.alert(f"{item['key']} (Task {item['task_id']}) was stopped outside the scheduler; "
                               "not repeated (use `requeue`).")
            else:  # CREATED / INIT: started, but not running any more (e.g. the core restarted)
                self.failure(item, f"Task is not running (state {STATE_NAMES.get(state, state)}); core restarted?",
                             info)
        self.db.commit()

    def finish(self, item, status: str, reason: str | None, info: dict | None = None):
        now = self.now()
        result = self.collect_result(item["task_id"], info)
        self.db.execute("UPDATE items SET status = ?, finished_at = ?, reason = ?, result = ? WHERE key = ?",
                        (status, iso(now), reason, json.dumps(result), item["key"]))
        self.db.execute("UPDATE attempts SET status = ?, finished_at = ?, reason = ?, result = ? WHERE task_id = ?",
                        (status, iso(now), reason, json.dumps(result), item["task_id"]))
        if status == ATTRITION:
            self.alert(f"{item['key']} (Task {item['task_id']}) ended as attrition: {reason}", level="info")
        logging.info("%s %s: %s", item["key"], status, reason or "")

    def failure(self, item, reason: str, info: dict | None = None):
        """Infrastructure failure: repeat the item as a new Task, or halt after max_retries repetitions."""
        if item["attempt"] <= self.cfg["max_retries"]:
            self.finish(item, QUEUED, f"attempt {item['attempt']} failed: {reason}", info)
            self.db.execute("UPDATE attempts SET status = ? WHERE task_id = ?", (FAILED, item["task_id"]))
            self.alert(f"{item['key']} attempt {item['attempt']} failed (Task {item['task_id']}), it will be "
                       f"repeated: {reason}")
        else:
            self.finish(item, FAILED, f"attempt {item['attempt']} failed: {reason}", info)
            self.halt(f"{item['key']} failed {item['attempt']} times (last Task {item['task_id']}): {reason}")

    def collect_result(self, task_id: str, info: dict | None) -> dict:
        """Summary of a Task from its folder: valid iterations, best development score, token usage."""
        result = {
            "iterations": info.get("current_iteration") if info else None,
            "valid": info.get("iterations_valid") if info else None,
        }
        folder = os.path.join(self.out_task, f"t_{task_id}")
        scores = []
        for path in glob.glob(os.path.join(folder, "solution", "sol_*", "*.dat")):
            with contextlib.suppress(Exception):
                res = load_json(path).get("results") or {}
                mean = list((res.get("Mean") or {}).values())
                if mean and mean[0] is not None:
                    scores.append(float(mean[0]))
        result["best_score"] = min(scores) if scores else None
        usage = os.path.join(folder, "usage_iterations.csv")
        if os.path.exists(usage):
            with contextlib.suppress(Exception), open(usage, encoding="utf-8") as f:
                rows = list(csv.DictReader(f, delimiter=";"))
                total = lambda col: sum(int(float(r[col] or 0)) for r in rows)  # noqa: E731
                result.update({
                    "generator_input_tokens": total("generator_input_tokens"),
                    "generator_output_tokens": total("generator_output_tokens"),
                    "summarizer_input_tokens": total("other_input_tokens"),
                    "summarizer_output_tokens": total("other_output_tokens"),
                })
        return result

    def archive_finished(self):
        """Zip the folders of ended attempts and archive the Tasks in the core (retried on the next tick on error)."""
        rows = self.db.execute("SELECT * FROM attempts WHERE finished_at IS NOT NULL AND (archive IS NULL OR "
                               "archived_in_core = 0)").fetchall()
        if not rows:
            return
        os.makedirs(self.archive_dir, exist_ok=True)
        for a in rows:
            try:
                if a["archive"] is None:
                    folder = os.path.join(self.out_task, f"t_{a['task_id']}")
                    name = f"{a['key']}__a{a['attempt']}__{a['status']}__{a['task_id']}"
                    if os.path.isdir(folder):
                        tmp = shutil.make_archive(os.path.join(self.archive_dir, name + ".tmp"), "zip", folder)
                        os.replace(tmp, os.path.join(self.archive_dir, name + ".zip"))
                    else:
                        name = "(missing folder)"
                    self.db.execute("UPDATE attempts SET archive = ? WHERE task_id = ?", (name, a["task_id"]))
                    self.db.commit()
                if not a["archived_in_core"]:
                    with contextlib.suppress(requests.HTTPError):  # the Task may be gone already
                        self.core.archive(a["task_id"])
                    self.db.execute("UPDATE attempts SET archived_in_core = 1 WHERE task_id = ?", (a["task_id"],))
                    self.db.commit()
            except Exception as e:
                logging.exception("Archiving failed")
                self.alert(f"Archiving of {a['key']} (Task {a['task_id']}) failed: {e!r}")

    def fill(self):
        """Start queued items while fewer than max_concurrent Tasks run in the core."""
        running_core = sum(state_of(i) == RUN for i in self.core.states(None).values())
        slots = self.cfg["max_concurrent"] - running_core
        while slots > 0:
            item = self.db.execute("SELECT * FROM items WHERE status = ? ORDER BY seq LIMIT 1", (QUEUED,)).fetchone()
            if item is None:
                return
            try:
                task_id = self.start(item)
            except (Exception, SystemExit) as e:  # configuration problem (token, core): do not try the rest
                self.halt(f"{item['key']} could not be started: {e}")
                return
            logging.info("Started %s as Task %s", item["key"], task_id)
            slots -= 1

    def start(self, item) -> str:
        if self.tokens is None:
            raise RuntimeError("No LLM tokens (tick must be called with --tokens-stdin by experiment_scheduler.sh).")
        p = self.plan
        g = p["generators"][item["generator"]]
        s = p["scenarios"][item["scenario"]]
        args = types.SimpleNamespace(
            provider=g["provider"], model=g["model"], token=g.get("token"),
            summarizer_provider=s["provider"] if s else None,
            summarizer_model=s["model"] if s else None,
            summarizer_token=s.get("token") if s else None,
            function=p["functions"][item["function"]], time=p["time"], runs=p["runs"],
            iterations=p["iterations"], author=self.cfg["author"], name_prefix="")
        cell = p["cells"][item["cell"]]
        name = p["task_name"].format(generator=item["generator"], function=item["function"], cell=cell,
                                     scenario=item["scenario"], rep=item["rep"])
        config = create_smoke_tasks.task_config(args, cell, self.tokens, name=name)
        task_id = self.core.create(config)
        self.core.run(task_id)
        now = iso(self.now())
        attempt = item["attempt"] + 1
        self.db.execute(
            "UPDATE items SET status = ?, attempt = ?, task_id = ?, started_at = ?, finished_at = NULL, "
            "last_iteration = NULL, last_progress_at = ?, stuck_alerted = 0 WHERE key = ?",
            (RUNNING, attempt, task_id, now, now, item["key"]))
        self.db.execute("INSERT INTO attempts (task_id, key, attempt, started_at, status) VALUES (?, ?, ?, ?, ?)",
                        (task_id, item["key"], attempt, now, RUNNING))
        self.db.commit()
        return task_id

    # ---------------------------------------------------------------- mail
    def send_alerts(self):
        pending = self.db.execute("SELECT * FROM alerts WHERE sent_at IS NULL ORDER BY id").fetchall()
        if not pending:
            return
        last = parse_iso(self.meta("last_alert_mail"))
        urgent = any(a["level"] == "halt" for a in pending)
        if not urgent and last and self.now() - last < dt.timedelta(minutes=self.cfg["alert_interval_minutes"]):
            return
        if all(a["level"] == "info" for a in pending):
            return  # informational only, goes into the daily report
        body = "\n\n".join(f"[{a['created_at']}] {a['level'].upper()}: {a['text']}" for a in pending)
        subject = f"[{self.plan['name']}] {'HALTED' if urgent else 'Alert'}: {len(pending)} event(s)"
        self.mailer(self.cfg, subject, body + "\n\n" + self.status_text())  # on error: tried again next tick
        now = iso(self.now())
        self.db.execute(f"UPDATE alerts SET sent_at = ? WHERE id IN ({','.join('?' * len(pending))})",
                        (now, *[a["id"] for a in pending]))
        self.set_meta("last_alert_mail", now)

    def local_now(self) -> dt.datetime:
        try:
            from zoneinfo import ZoneInfo

            return self.now().astimezone(ZoneInfo(self.cfg["timezone"]))
        except Exception:
            return self.now().astimezone()

    def daily_report(self, force: bool = False):
        local = self.local_now()
        hh, mm = map(int, self.cfg["report_time"].split(":"))
        today = local.date().isoformat()
        if not force and (self.meta("last_report") == today or (local.hour, local.minute) < (hh, mm)):
            return
        since = self.now() - dt.timedelta(days=1)
        self.mailer(self.cfg, f"[{self.plan['name']}] Daily report {today}", self.status_text(since=since))
        self.set_meta("last_report", today)

    # ---------------------------------------------------------------- status
    def status_text(self, since: dt.datetime | None = None) -> str:
        now = self.now()
        lines = []
        halted, paused = self.meta("halted"), self.meta("paused")
        lines.append(f"Scheduler: {'HALTED - ' + halted if halted else 'paused' if paused else 'active'}")
        lines.append(f"Time: {self.local_now():%Y-%m-%d %H:%M %Z}")
        rows = self.db.execute("SELECT * FROM items ORDER BY seq").fetchall()
        if not rows:
            return "\n".join(lines + ["", "No work enqueued."])

        lines += ["", "Batches (done / attrition / failed / stopped / running / queued / total):"]
        for batch in dict.fromkeys(r["batch"] for r in rows):
            b = [r for r in rows if r["batch"] == batch]
            c = {s: sum(r["status"] == s for r in b) for s in (DONE, ATTRITION, FAILED, STOPPED, RUNNING, QUEUED)}
            lines.append(f"  {batch}: {c[DONE]} / {c[ATTRITION]} / {c[FAILED]} / {c[STOPPED]} / {c[RUNNING]} / "
                         f"{c[QUEUED]} / {len(b)}")

        running = [r for r in rows if r["status"] == RUNNING]
        lines += ["", f"Running ({len(running)}):"]
        for r in running:
            started = parse_iso(r["started_at"])
            progress = parse_iso(r["last_progress_at"])
            lines.append(f"  {r['key']} attempt {r['attempt']} Task {r['task_id']} iteration {r['last_iteration']} "
                         f"running {hours(now - started)}, last progress {hours(now - progress)} ago")

        done = [r for r in rows if r["status"] == DONE and r["started_at"] and r["finished_at"]]
        queued = sum(r["status"] == QUEUED for r in rows)
        if done and (queued or running):
            avg = sum((parse_iso(r["finished_at"]) - parse_iso(r["started_at"])).total_seconds() for r in done)
            avg /= len(done)
            eta = avg * (queued + len(running)) / self.cfg["max_concurrent"]
            lines.append(f"\nAverage Task duration {avg / 3600:.1f} h; estimated time to finish the queue "
                         f"{eta / 3600:.0f} h ({eta / 86400:.1f} days).")

        lines += ["", "Configurations (done/total repetitions, + attrition, ! failed):"]
        configs = {}
        for r in rows:
            configs.setdefault(r["key"].rsplit("-r", 1)[0], []).append(r["status"])
        for key, st in configs.items():
            if all(s == QUEUED for s in st):
                continue
            extra = (f" +{st.count(ATTRITION)}" if ATTRITION in st else "") + (
                f" !{st.count(FAILED)}" if FAILED in st else "")
            lines.append(f"  {key} ({self.plan['cells'][key.split('-')[2]]}): {st.count(DONE)}/{len(st)}{extra}")

        attempts = self.db.execute("SELECT * FROM attempts WHERE finished_at IS NOT NULL").fetchall()
        tokens = {}
        for a in attempts:
            res = json.loads(a["result"] or "{}")
            g = a["key"].split("-")[0]
            for k in ("generator_input_tokens", "generator_output_tokens", "summarizer_input_tokens",
                      "summarizer_output_tokens"):
                tokens.setdefault(g, {}).setdefault(k, 0)
                tokens[g][k] += res.get(k) or 0
        if tokens:
            lines += ["", "Billed tokens of ended attempts (generator in/out, summarizer in/out):"]
            for g, t in sorted(tokens.items()):
                lines.append(f"  {g}: {t['generator_input_tokens']:,} / {t['generator_output_tokens']:,}; "
                             f"{t['summarizer_input_tokens']:,} / {t['summarizer_output_tokens']:,}")

        if since is not None:
            recent = [a for a in attempts if parse_iso(a["finished_at"]) >= since]
            lines += ["", f"Ended in the last 24 h ({len(recent)}):"]
            for a in recent:
                res = json.loads(a["result"] or "{}")
                score = f"{res['best_score']:.6e}" if res.get("best_score") is not None else "-"
                lines.append(f"  {a['key']} attempt {a['attempt']}: {a['status']}, valid {res.get('valid')}, "
                             f"best {score}" + (f" - {a['reason']}" if a["reason"] else ""))
            alerts = self.db.execute("SELECT * FROM alerts WHERE created_at >= ? ORDER BY id",
                                     (iso(since),)).fetchall()
            if alerts:
                lines += ["", f"Events in the last 24 h ({len(alerts)}):"]
                lines += [f"  [{a['created_at']}] {a['level']}: {a['text']}" for a in alerts]
        return "\n".join(lines)


####################################################################
#########  CLI
####################################################################
def main(argv=None):
    parser = argparse.ArgumentParser(description="Scheduler of the context-summarizer experiment.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("enqueue", help="Add a batch of work (items not added yet).")
    p.add_argument("--batch", required=True, help="Name of the batch, e.g. G1-phase1.")
    p.add_argument("--generators", nargs="+", required=True, help="G1 G2 G3")
    p.add_argument("--functions", nargs="+", help="F1 F2 F3 (default all)")
    p.add_argument("--cells", nargs="+", help="C1 ... C9 (default all)")
    p.add_argument("--scenarios", nargs="+", help="S1 S2 (default both)")
    p.add_argument("--reps", help="Repetitions, e.g. 1-11 or 1,2,5 (default all)")
    p = sub.add_parser("dequeue", help="Remove items that have not started yet.")
    p.add_argument("--batch")
    p.add_argument("--keys", nargs="+")
    p = sub.add_parser("tick", help="One scheduler step (run by cron).")
    p.add_argument("--tokens-stdin", action="store_true", help="Read the frontEASE LLM tokens from stdin.")
    sub.add_parser("status", help="Print the state.")
    sub.add_parser("pause", help="Do not start new Tasks (running ones continue).")
    sub.add_parser("resume", help="Start new Tasks again (also after a halt).")
    p = sub.add_parser("requeue", help="Queue ended items again (failed, attrition, stopped).")
    p.add_argument("keys", nargs="+")
    p = sub.add_parser("report", help="Print the daily report; --send mails it now.")
    p.add_argument("--send", action="store_true")
    sub.add_parser("test-email", help="Send a test e-mail.")
    p = sub.add_parser("config-get", help="Print one configuration value (used by experiment_scheduler.sh).")
    p.add_argument("key")
    args = parser.parse_args(argv)

    os.makedirs(EXP_DIR, exist_ok=True)
    if args.cmd == "config-get":
        print(load_config(EXP_DIR).get(args.key, ""))
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(os.path.join(EXP_DIR, "scheduler.log"), encoding="utf-8"),
                                  logging.StreamHandler()])
    s = Scheduler()
    if args.cmd == "enqueue":
        reps = parse_reps(args.reps, s.plan["repetitions"]) if args.reps else None
        n = s.enqueue(args.batch, args.generators, args.functions, args.cells, args.scenarios, reps)
        print(f"Enqueued {n} items into batch {args.batch}.")
    elif args.cmd == "dequeue":
        if not args.batch and not args.keys:
            sys.exit("Use --batch or --keys.")
        print(f"Removed {s.dequeue(args.batch, args.keys)} queued items.")
    elif args.cmd == "tick":
        if args.tokens_stdin:
            s.tokens = create_smoke_tasks.read_frontease_tokens(sys.stdin)
        s.tick()
    elif args.cmd == "status":
        print(s.status_text(since=s.now() - dt.timedelta(days=1)))
    elif args.cmd == "pause":
        s.set_meta("paused", iso(s.now()))
        print("Paused: no new Tasks are started.")
    elif args.cmd == "resume":
        s.set_meta("paused", None)
        s.set_meta("halted", None)
        print("Resumed.")
    elif args.cmd == "requeue":
        print(f"Requeued {s.requeue(args.keys)} items.")
    elif args.cmd == "report":
        text = s.status_text(since=s.now() - dt.timedelta(days=1))
        print(text)
        if args.send:
            print("sent" if s.mailer(s.cfg, f"[{s.plan['name']}] Report", text) else "mail not configured")
    elif args.cmd == "test-email":
        print("sent" if s.mailer(s.cfg, f"[{s.plan['name']}] Test e-mail", "The scheduler can send e-mails.")
              else "mail not configured (smtp.host / recipients)")


if __name__ == "__main__":
    main()
