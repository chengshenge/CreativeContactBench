"""Import the exact private author handoff without downloading or publishing it."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,zipfile
EXPECTED='793b7a4e545ffa2184fdba1dcf787b3285eb7a36fa347c8af1166844e6daed5c'
p=argparse.ArgumentParser();p.add_argument('--zip',required=True,type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
assert hashlib.sha256(args.zip.read_bytes()).hexdigest()==EXPECTED,'Wrong archive or changed bytes; obtain the pinned author handoff'
assert not args.output.exists(),'Destination exists; choose a new directory'
with zipfile.ZipFile(args.zip) as z:
 for item in z.infolist():
  name=PurePosixPath(item.filename)
  assert not name.is_absolute() and '..' not in name.parts and '\\' not in item.filename
  assert name.parts[0]=='rgb30_bf16_handoff_20261017'
 z.extractall(args.output)
root=args.output/'rgb30_bf16_handoff_20261017'
manifest=json.loads((root/'bundle_manifest.json').read_text())
for name,digest in manifest['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
print('Imported verified author bundle:',root)
print('Read:',root/'CODEX_START_HERE.md')
