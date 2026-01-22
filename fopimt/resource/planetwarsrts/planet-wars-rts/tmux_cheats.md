# 🧠 tmux Cheat Sheet

A quick reference for managing background terminal sessions with `tmux`.

---

## 🚀 Start a New Session

```bash
tmux new -s mysession
```

Creates and enters a new tmux session named `mysession`.

---

## 🔄 Detach from a Session

While inside tmux, press:

```
Ctrl + b, then d
```

This detaches the session but keeps it running in the background.

---

## 🔙 Reattach to a Session

```bash
tmux attach -t mysession
```

If there's only one session:

```bash
tmux attach
```

---

## 🔎 List All Sessions

```bash
tmux ls
```

Displays all currently running tmux sessions.

---

## ❌ Kill a Session

```bash
tmux kill-session -t mysession
```

Or, from inside a tmux session:

```bash
exit
```

---

## 📌 Bonus: Common Session Names

- `tmux new -s bot` – start a background bot
- `tmux attach -t bot` – reattach to bot session

---

Happy multiplexing! 🎛️
