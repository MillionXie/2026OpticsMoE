"""Launch one measured T08 stage on the bench PC interactive desktop."""
import argparse
import base64
from pathlib import Path

import paramiko

HERE = Path(__file__).resolve().parent
STAGES = ('vision_router', 'vision_expert', 'vision_global',
          'language_router', 'language_expert', 'language_global')
BASE = r'E:\code\guest\2026OpticsMoE\ABO_T2I_10cm_alpha040_20260925'
PYTHON = r'E:\code\guest\2026OpticsMoE\ABO_Lab_SHS_8um\.venv_gpu\Scripts\python.exe'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=STAGES, required=True)
    p.add_argument('--password', required=True)
    p.add_argument('--exposure-us', type=float, default=10000)
    p.add_argument('--wait-ms', type=float, default=240)
    a = p.parse_args()
    index = STAGES.index(a.stage)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect('1.tcp.vip.cpolar.top', port=12705, username='PS', password=a.password, timeout=15)
    try:
        sftp = c.open_sftp()
        sftp.put(str(HERE / 'capture_full_stage.py'), BASE.replace('\\', '/') + '/capture_full_stage.py')
        sftp.close()
        task = 'ABO_T2I_Alpha040_' + a.stage
        script = BASE + r'\capture_full_stage.py'
        log = BASE + fr'\full_test\{index+1:02d}_{a.stage}\capture.log'
        command = (f'"{PYTHON}" -u "{script}" --stage {a.stage} '
                   f'--exposure-us {a.exposure_us:g} --wait-ms {a.wait_ms:g}')
        ps = f'''$ErrorActionPreference='Stop'
$sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$xml=@"
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
<Principals><Principal id="Author"><UserId>$sid</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
<Settings><ExecutionTimeLimit>PT6H</ExecutionTimeLimit></Settings>
<Actions Context="Author"><Exec><Command>cmd.exe</Command><Arguments>/c &quot;{command} &gt; &quot;&quot;{log}&quot;&quot; 2&gt;&amp;1&quot;</Arguments></Exec></Actions>
</Task>
"@
Register-ScheduledTask -TaskName '{task}' -Xml $xml | Out-Null
Start-ScheduledTask -TaskName '{task}'
Get-ScheduledTask -TaskName '{task}' | Select-Object TaskName,State | ConvertTo-Json -Compress
'''
        encoded = base64.b64encode(ps.encode('utf-16le')).decode('ascii')
        _, stdout, stderr = c.exec_command('powershell -NoProfile -EncodedCommand ' + encoded)
        print(stdout.read().decode(errors='replace'))
        print(ascii(stderr.read().decode(errors='replace')))
    finally:
        c.close()


if __name__ == '__main__':
    main()
