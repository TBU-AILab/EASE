#!/bin/sh
# One step of the experiment scheduler (tools/experiment_scheduler.py tick), meant for cron on the server, e.g.
#
#   */10 * * * * /PaperSummarizer/frontEASE/src/FoP_IMT.Core/tools/experiment_scheduler.sh >> /PaperSummarizer/frontEASE/src/FoP_IMT.Core/experiment/cron.log 2>&1
#
# 1) reads the LLM tokens of the frontEASE user `author` (experiment/scheduler.json) from the frontEASE database and
#    pipes them to the tick (never on a command line),
# 2) moves the zipped Task folders from experiment/archive to `archive_target` (e.g. a NAS mount), if configured;
#    a zip is removed only after its copy was verified.
set -eu

cd "$(dirname "$0")/../../.."  # frontEASE directory (docker-compose.yml)
CORE=src/FoP_IMT.Core

conf() {
    docker compose exec -T backend-core python tools/experiment_scheduler.py config-get "$1" | tr -d '\r'
}

EMAIL=$(conf author)
if [ -z "$EMAIL" ]; then
    echo "$(date -Is) experiment/scheduler.json: 'author' is not set" >&2
    exit 1
fi

docker compose exec -T postgres sh -c \
    'psql -U "$POSTGRES_USER" -d FrontEASE -X -q -A -t -F "$(printf "\t")" -v ON_ERROR_STOP=1 -v email="$1" -f -' \
    sh "$EMAIL" < "$CORE/tools/frontease_tokens.sql" |
    docker compose exec -T backend-core python tools/experiment_scheduler.py tick --tokens-stdin

TARGET=$(conf archive_target)
if [ -n "$TARGET" ]; then
    mkdir -p "$TARGET"
    for f in "$CORE"/experiment/archive/*.zip; do
        [ -e "$f" ] || continue
        dest="$TARGET/$(basename "$f")"
        if cp "$f" "$dest.part" && cmp -s "$f" "$dest.part" && mv "$dest.part" "$dest"; then
            rm "$f"
        else
            echo "$(date -Is) copy of $f to $TARGET failed, kept" >&2
            rm -f "$dest.part"
        fi
    done
fi
