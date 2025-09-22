---
metadata:
  source: Okta Support Documentation
  product: Okta Verify MFA
  version: 2.15
  date: 2025-09-22
  doc_type: Official Troubleshooting Guide
scenario: okta_mfa_failure
---

# Okta Verify Push Timeout Troubleshooting

## Problem Description
Users experience timeouts when attempting to authenticate via Okta Verify push notifications, blocking access to applications or RDP sessions. Common in mobile-heavy remote work setups.

**Symptoms:**
- No push notification received on Okta Verify app.
- Error: "Push notification timed out" or "Authentication failed."
- Intermittent issues after device OS updates.

## Possible Causes
- Unstable network (WiFi/mobile data) preventing push delivery.
- Device clock out of sync with Okta servers.
- App permissions disabled (notifications, background refresh).
- Okta server-side issues or factor enrollment errors.

## Ground Truth Resolution Steps
1. **Verify Network and App:**
   - Ensure device has stable internet; toggle WiFi/mobile data.
   - Update Okta Verify to latest version; force close and reopen.
   - Check permissions: Enable notifications and background refresh in device settings (iOS/Android).

2. **Sync Device Clock:**
   - iOS: Settings > General > Date & Time > Set Automatically.
   - Android: Settings > System > Date & Time > Use network-provided time.
   - Restart device and retry push.

3. **User-Side Checks:**
   - Verify correct account selected in Okta Verify.
   - Test push on alternate device if available.

4. **Admin-Side Verification:**
   - Okta Admin Console: Directory > People > User > Security > Factors > Reset push factor if enrolled incorrectly.
   - Check System Log: Reports > System Log > Filter by user/event time for errors (e.g., "Push delivery failed").
   - Re-enroll factor if needed: User > Security > MFA > Add Factor.

5. **Escalation:**
   - If persists, collect Okta Verify debug logs (App Settings > Enable Debug) and escalate to Okta Support.

## References
- Okta Help Center: Push Notification Troubleshooting.
- Expected Outcome: Push notification succeeds in <5min post-sync.

**KPI Impact:** Time-to-first-action: 3min; MTTR reduction: 50%; % helpful sources in top-5: 85%.