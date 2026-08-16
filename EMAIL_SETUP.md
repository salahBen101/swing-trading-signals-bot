# Getting signals emailed to you

Your address is already configured (`salaheddine.bensassi@gmail.com`, in `.env`). One field is
missing, and only you can supply it.

## Why you have to do this part

Gmail will not accept your normal account password from a program — it requires an **App
Password**, a separate 16-character credential scoped to one application and revocable on its
own. That is a good thing: it means this scanner can send mail without ever holding the keys
to your Google account.

It also means the credential goes straight from Google into your local `.env` file. It is not
something to paste into a chat, and nothing in this project prints or logs it.

## Step 1 — create the App Password (about 2 minutes)

1. Go to **myaccount.google.com/security**
2. Turn on **2-Step Verification** if it is not already on (App Passwords require it)
3. Search that page for **App passwords**, or go directly to
   **myaccount.google.com/apppasswords**
4. Create one — name it anything, e.g. `RSI2 scanner`
5. Google shows 16 characters like `abcd efgh ijkl mnop`. Copy them.

## Step 2 — paste it in

Open `.env` in this folder and put it after the `=`:

```
NOTIFY_EMAIL_PASSWORD=abcdefghijklmnop
```

Spaces are fine either way. Save the file.

`.env` is gitignored, so it never leaves your machine.

## Step 3 — confirm it works

```bash
.venv\Scripts\python.exe scanner\rsi2_scanner.py --test-notify
```

You should see `email: sent` and a test message in your inbox. If it says `email failed:
SMTPAuthenticationError`, the password was rejected — usually because it is the account
password rather than an App Password.

## Step 4 — make it run itself at 15:30

```bash
powershell -ExecutionPolicy Bypass -File .\setup_daily_scan.ps1
```

That registers a Windows Scheduled Task for weekdays at 15:30 Eastern. It works out your
machine's offset from ET and adjusts, so it fires at the right moment even if you are not in a
US timezone. It prints the local time it chose — worth a glance to confirm.

Check it immediately with:

```bash
powershell -Command "Start-ScheduledTask -TaskName 'RSI2 Scanner'"
```

Remove it any time:

```bash
powershell -Command "Unregister-ScheduledTask -TaskName 'RSI2 Scanner' -Confirm:$false"
```

## What arrives

Only when something actually triggers — no daily "nothing today" mail. Each signal email
carries the ticker and price, the RSI(2) reading, how far above its 200-day average it is, the
close that would cancel the signal if the market is still open, and the historical exit rule.

The options panel appears in the terminal output rather than the email, since chain data is
long and goes stale within minutes.
