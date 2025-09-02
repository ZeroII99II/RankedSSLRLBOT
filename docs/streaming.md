# Streaming Training and RLBot Gameplay

This guide covers how to run continuous training/export and display the policy in Rocket League, then broadcast to Twitch using OBS.

## Continuous Training and Export

The script `scripts/night_train_v2.sh` loops forever, training the bot and exporting a TorchScript policy on each iteration.

```bash
scripts/night_train_v2.sh
```

- Training runs for 1,000,000 steps with 1 environment.
- If training and export succeed, the policy is copied to `models/exported/ssl_policy.ts`.
- On failure, the script waits 5 seconds and restarts.

## Displaying the Policy in Rocket League

Launch RLBot locally to watch the latest exported policy.

```bash
scripts/run_rlbot_local.sh
```

- RLBot looks for the policy at `models/exported/ssl_policy.ts`. Set `SSL_POLICY_PATH` to override this location.
- Ensure Rocket League is installed and open; the RLBot client will attach and control cars.

## Streaming to Twitch

Use [OBS Studio](https://obsproject.com/) or similar software:

1. Open OBS and add a **Window Capture** source for the Rocket League window.
2. In **Settings → Stream**, select **Twitch** and paste your stream key.
3. Optional: add microphone or overlay sources.
4. Click **Start Streaming** to broadcast the RLBot match.

With training running in one terminal and RLBot in another, OBS will capture the gameplay and send it to your Twitch channel.
