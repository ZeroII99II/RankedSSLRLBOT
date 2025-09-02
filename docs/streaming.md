# Streaming Training and RLBot Gameplay

This guide shows how to watch the trainer, visualise the policy in Rocket League and broadcast the window to Twitch.

## Watch the Trainer

Use the `--render` flag to open a lightweight viewer while PPO training runs in the background:

```bash
python src/training/train_v2.py --render
```

The viewer uses a single non-vectorised environment; training continues with the number of environments specified by `--envs`.

## Display the Policy in Rocket League

After exporting a policy to `models/exported/ssl_policy.ts`, launch RLBot locally to watch the bot control cars in the game client:

```bash
scripts/run_rlbot_local.sh
```

Set the `SSL_POLICY_PATH` environment variable if the policy lives elsewhere.

## Stream to Twitch

1. Open [OBS Studio](https://obsproject.com/).
2. Add a **Window Capture** source for either the training viewer or the Rocket League window.
3. Under **Settings → Stream**, choose **Twitch** and enter your stream key.
4. Click **Start Streaming** to broadcast.
