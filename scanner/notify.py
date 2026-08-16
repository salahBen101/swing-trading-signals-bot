"""Desktop and email notification for scanner signals.

Credentials are read from a local .env file and never appear in code, output or logs. For
Gmail you need an App Password (Google account -> Security -> 2-Step Verification -> App
passwords), not your normal password: an app password is scoped to one application and can be
revoked on its own.

Nothing here is required for the scanner to run. If notifications are not configured the
scanner simply prints to the console as usual.
"""

from __future__ import annotations

import os
import smtplib
import subprocess
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path | None = None) -> dict[str, str]:
    """Minimal .env reader - avoids a dependency and avoids exporting secrets process-wide."""
    env_path = path or PROJECT_ROOT / ".env"
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass
class NotifyConfig:
    desktop: bool
    email_enabled: bool
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    email_to: str

    @classmethod
    def from_env(cls) -> "NotifyConfig":
        env = {**load_env(), **os.environ}
        user = env.get("NOTIFY_EMAIL_USER", "")
        password = env.get("NOTIFY_EMAIL_PASSWORD", "")
        to = env.get("NOTIFY_EMAIL_TO", user)
        return cls(
            desktop=env.get("NOTIFY_DESKTOP", "true").lower() == "true",
            email_enabled=bool(user and password and to),
            smtp_host=env.get("NOTIFY_SMTP_HOST", "smtp.gmail.com"),
            smtp_port=int(env.get("NOTIFY_SMTP_PORT", "587")),
            smtp_user=user,
            smtp_password=password,
            email_to=to,
        )

    def describe(self) -> str:
        bits = []
        bits.append("desktop ON" if self.desktop else "desktop off")
        if self.email_enabled:
            # Show only enough of the address to confirm it is the right one.
            local, _, domain = self.email_to.partition("@")
            masked = f"{local[:2]}***@{domain}" if domain else "configured"
            bits.append(f"email -> {masked}")
        else:
            bits.append("email off (set NOTIFY_EMAIL_* in .env)")
        return ", ".join(bits)


def send_desktop(title: str, message: str) -> bool:
    """Windows toast via PowerShell. No third-party dependency, no install step."""
    safe_title = title.replace("'", "''")
    safe_message = message.replace("'", "''")
    script = f"""
$ErrorActionPreference = 'Stop'
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null
$template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent(
    [Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$texts = $template.GetElementsByTagName('text')
$texts.Item(0).AppendChild($template.CreateTextNode('{safe_title}')) | Out-Null
$texts.Item(1).AppendChild($template.CreateTextNode('{safe_message}')) | Out-Null
$toast = [Windows.UI.Notifications.ToastNotification]::new($template)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('RSI2 Scanner').Show($toast)
"""
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            check=True,
            capture_output=True,
            timeout=20,
        )
        return True
    except Exception:
        # Fall back to a message box, which works on older Windows without the toast API.
        try:
            fallback = (
                "Add-Type -AssemblyName PresentationFramework; "
                f"[System.Windows.MessageBox]::Show('{safe_message}','{safe_title}')"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", fallback],
                check=True,
                capture_output=True,
                timeout=20,
            )
            return True
        except Exception:
            return False


def send_email(config: NotifyConfig, subject: str, body: str) -> bool:
    if not config.email_enabled:
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = config.smtp_user
    msg["To"] = config.email_to
    msg.set_content(body)
    try:
        with smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=30) as server:
            server.starttls()
            server.login(config.smtp_user, config.smtp_password)
            server.send_message(msg)
        return True
    except Exception as exc:
        # Never echo the exception verbatim - SMTP errors can contain the credential.
        print(f"    email failed: {type(exc).__name__}")
        return False


def build_message(signals: list, provisional: bool) -> tuple[str, str]:
    tickers = ", ".join(s.ticker for s in signals)
    subject = f"RSI(2) signal: {tickers}"

    lines = [f"{len(signals)} signal(s): {tickers}", ""]
    if provisional:
        lines += [
            "PROVISIONAL - the market is still open and the daily bar is not final.",
            "Each signal below shows the close that would turn it off.",
            "",
        ]
    try:
        from universe import sector_of, tier_of
    except ImportError:
        from scanner.universe import sector_of, tier_of

    for s in signals:
        lines.append(f"{s.ticker} @ {s.price:,.2f}  ({s.pct_from_prev:+.2f}% today)")
        lines.append(f"   sector: {sector_of(s.ticker)}   evidence: {tier_of(s.ticker).upper()}")
        lines.append(f"   RSI(2) {s.rsi2:.2f}   200-day avg {s.sma200:,.2f} "
                     f"({(s.price / s.sma200 - 1) * 100:+.1f}% above)")
        if provisional and s.flip_price is not None:
            lines.append(f"   turns off on a close above {s.flip_price:,.2f}")
        lines.append("")

    lines += [
        "Historical exit rule: RSI(2) > 65, or a close above the 5-day average,",
        "typically 3-4 sessions.",
        "",
        "Validation: 7,291 trades / 120 names post-2009, median +0.50% per trade,",
        "70.4% win rate. The same rules produced roughly zero on SPY 2010-2019.",
        "",
        "This is an indicator reading, not advice. No orders have been placed.",
    ]
    return subject, "\n".join(lines)


def notify(signals: list, provisional: bool, config: NotifyConfig | None = None) -> None:
    if not signals:
        return
    config = config or NotifyConfig.from_env()
    subject, body = build_message(signals, provisional)

    print(f"\n  Notifications ({config.describe()}):")
    if config.desktop:
        summary = ", ".join(f"{s.ticker} RSI {s.rsi2:.1f}" for s in signals[:4])
        ok = send_desktop(f"{len(signals)} RSI(2) signal(s)", summary)
        print(f"    desktop: {'sent' if ok else 'failed'}")
    if config.email_enabled:
        ok = send_email(config, subject, body)
        print(f"    email:   {'sent' if ok else 'failed'}")
