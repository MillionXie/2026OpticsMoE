"""Launch pinned full-test capture in the bench PC's interactive desktop session."""
import argparse
import base64
from pathlib import Path

import paramiko

HERE = Path(__file__).resolve().parent
HOST, PORT, USER = '1.tcp.vip.cpolar.top', 12705, 'PS'
REMOTE_DIR = r'E:\code\guest\2026OpticsMoE\ABO_T2I_10cm_alpha040_20260925'
PYTHON = r'E:\code\guest\2026OpticsMoE\ABO_Lab_SHS_8um\.venv_gpu\Scripts\python.exe'
TASK = 'ABO_T2I_10cm_Alpha040_FullRouter'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--password', required=True)
    a = p.parse_args()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=a.password, timeout=15)
    try:
        sftp = client.open_sftp()
        sftp.put(str(HERE / 'capture_full_vision_router.py'),
                 REMOTE_DIR.replace('\\', '/') + '/capture_full_vision_router.py')
        sftp.close()
        script = REMOTE_DIR + r'\capture_full_vision_router.py'
        log = REMOTE_DIR + r'\full_test\01_vision_router\capture.log'
        command = f'"{PYTHON}" -u "{script}"'
        ps = f'''$ErrorActionPreference='Stop'
$sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$xml=@"
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
<Principals><Principal id="Author"><UserId>$sid</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
<Settings><ExecutionTimeLimit>PT6H</ExecutionTimeLimit></Settings>
<Actions Context="Author"><Exec><Command>cmd.exe</Command><Arguments>/c &quot;{command} &gt; &quot;&quot;{log}&quot;&quot; 2&gt;&amp;1&quot;</Arguments></Exec></Actions>
</Task>
"@
Register-ScheduledTask -TaskName '{TASK}' -Xml $xml | Out-Null
Start-ScheduledTask -TaskName '{TASK}'
Get-ScheduledTask -TaskName '{TASK}' | Select-Object TaskName,State | ConvertTo-Json -Compress
'''
        encoded = base64.b64encode(ps.encode('utf-16le')).decode('ascii')
        _, stdout, stderr = client.exec_command('powershell -NoProfile -EncodedCommand ' + encoded)
        print(stdout.read().decode(errors='replace'))
        print(ascii(stderr.read().decode(errors='replace')))
    finally:
        client.close()


if __name__ == '__main__':
    main()
