# 8. Demo & Project Brief

> **One line:** Judges must *see* the agent think. The Mind View dashboard and a clear 2-page brief turn our engineering into a story.

---

## 8.1 Mind View dashboard

A screen that shows the game and the agent's "thoughts" side by side.

```text
┌──────────────────────────────┬───────────────────────────┐
│                              │   🗺️ MIND MAP              │
│     🎮 GAME VIEW              │   (what the agent has     │
│     (what the player sees)   │    drawn in its head)     │
│                              │   ● = me   ··· = visited   │
│                              │   🟥 = red locked door      │
├──────────────────────────────┼───────────────────────────┤
│ 🎯 CURRENT GOAL: Go to landmark → red door (12 m, 40° ←)  │
├──────────────────────────────┬───────────────────────────┤
│ 📒 NOTEBOOK                   │ 👁️ ATTENTION               │
│ ✅ red key (picked up)        │ Imp ███████ 0.71           │
│ 🟥 red door  12 m  ←          │ Zombie ██ 0.18             │
│ 🚪 door      5 m   →          │ Medkit █ 0.11              │
├──────────────────────────────┴───────────────────────────┤
│ ❤️ 64  🛡️ 20  🔫 38   🔑 red   │ Step 412/1050 │ 14 ms/step │
└──────────────────────────────────────────────────────────┘
```

**How it works:** the brain can return an optional `debug` dictionary (mind map, notebook, goal, attention weights). The dashboard reads it. On game day the debug output is **switched off** to save time.

**Build options:** an OpenCV window (simplest), or a small local web page. Both can record to MP4 for the backup video.

## 8.2 The 2-page brief (outline)

The rules ask for: **USP, neural architecture, spatial memory design, training/exploration pipeline.**

### Page 1
1. **Title + one-line pitch:** *"AK-Doom / Smriti: a DOOM agent that draws its own map and remembers where the locked door was."*
2. **USP (3 bullets):** mind map + landmark notebook, manager/worker, curiosity + curriculum.
3. **Architecture diagram:** the "inside the brain" picture from [Architecture §2.3](02-SYSTEM-ARCHITECTURE.md#23-inside-the-brain), simplified.
4. **Encoders:** circular ray conv, entity attention (handles any number of enemies), vitals MLP.
5. **Spatial memory design:** odometer → mind map → rotated crop; notebook slots + attention; GRU.

### Page 2
6. **Training pipeline:** our recurrent PPO, parallel games, reward table (short), curiosity.
7. **Exploration & generalisation:** curriculum stages, procedural maps, randomisation, train/val/test split.
8. **Results:** Mini-League table per tier + **ablation bar chart**.
9. **Speed:** average ms per decision, model size.
10. **Limitations & next steps** (being honest earns trust).

## 8.3 Live demo script (~5 minutes)

| Time | Show | Say |
|------|------|-----|
| 0:00 | Title slide | The problem: bots memorise, and we built one that remembers *and* understands |
| 0:30 | Tier 1 map, Mind View | Attention picks the most dangerous enemy |
| 1:30 | Tier 3 map | It sees the red door → notebook entry → explores → gets the key → goal switches to "go to landmark" → unlocks it |
| 3:00 | Ablation chart | Each idea adds points: remove the notebook and Tier 3 collapses |
| 3:45 | Speed numbers | ~X ms per decision on 1 CPU thread |
| 4:15 | A failure case | What still goes wrong, and how we'd fix it |
| 4:45 | Wrap-up | USP in one sentence |

**Backup:** pre-recorded videos of the same maps in case the live run misbehaves.

**Note:** the scorer records a replay of every run, and organisers watch the top teams' replays during manual review. Silly behaviour (spinning, wall-hugging) will be seen, so we check our own videos first.

## 8.4 Q&A preparation

Every team member should be able to answer these:

| Likely question | Short answer |
|-----------------|--------------|
| How does it remember without coordinates? | The odometer adds up our own movement (velocity in the player's frame, with Doom's friction modelled) and turns (predicted from our own turn buttons; measured on `two_rooms.wad`, re-check on larger maps). Drift was about 0.3% of the distance travelled there. The mind map is anchored at the spawn point, not the world. |
| How do you handle many enemies? | Attention over the 32 entity slots works with any count, and empty slots are masked. |
| Isn't the mind map a hardcoded rule? | No. RULES.md allows "preprocessing and encoding observations" and "learned memory modules". Every *decision* comes from the trained network, and there's no pathfinding or BFS. |
| How does it find the exit if rays can't see it? | A small RGB encoder, trained with an "is the exit in view?" helper task. |
| Why PPO? Did you write it? | It's stable and simple, and we wrote it ourselves to add the manager, curiosity and the asymmetric critic. |
| How do you avoid getting stuck? | Novelty reward + stuck penalty + visited breadcrumbs in the mind map. |
| How do you know it generalises? | Held-out validation and test maps, never trained on; randomised spawns, seeds and textures. |
| What did training-time cheats do? | Infinite ammo and auto-doors only in early stages; the last stages use the normal game, like evaluation. |
| What's its weakest point? | (Be honest, from our failure videos.) |
| How fast is it? | X ms average on 1 thread; the budget is 40 ms. |
