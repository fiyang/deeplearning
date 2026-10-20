"""Bounded four-connection HTTPS range download, reusing a verified-length prefix.
Final Git LFS SHA256 must match before the archive is published.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import re
import shutil
import time
import urllib.request
import urllib.error
from download import checksum


def fetch_range(url,path,start,end,total):
    path=Path(path);size=end-start+1
    if path.exists() and path.stat().st_size==size:return size
    if not url.startswith('https://'):raise ValueError('HTTPS required')
    temp=path.with_suffix('.partial')
    for attempt in range(5):
        try:
            req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(req,timeout=45) as r:
                expected=f'bytes {start}-{end}/{total}'
                if r.status!=206 or r.headers.get('Content-Range')!=expected:raise ValueError('Range response mismatch')
                count=0
                with temp.open('wb') as f:
                    while b:=r.read(2*1024*1024):
                        count+=len(b)
                        if count>size:raise ValueError('Range exceeds expected bytes')
                        f.write(b)
                if count!=size:raise OSError('Incomplete range')
            temp.replace(path);return size
        except urllib.error.HTTPError as e:
            if e.code in (401,403):raise RuntimeError(f'Download authorization denied HTTP {e.code}') from None
        except (urllib.error.URLError,TimeoutError,OSError):pass
        print(f'RANGE_RETRY {start}-{end} attempt={attempt+1}',flush=True)
        time.sleep(min(attempt+1,5))
    raise RuntimeError('Range retries exhausted; completed ranges retained')


def main():
    p=argparse.ArgumentParser();p.add_argument('--urls',type=Path,required=True);p.add_argument('--archive',required=True)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--manifest',type=Path,default=Path(__file__).with_name('source_manifest.json'));a=p.parse_args()
    info=next(f for f in json.loads(a.manifest.read_text())['files'] if f['name']==a.archive)
    target=a.directory/a.archive;prefix=target.with_name(target.name+'.part');total=info['bytes']
    if target.exists():
        if target.stat().st_size!=total or checksum(target)!=info['sha256']:raise ValueError('Existing archive verification failed')
        print('ARCHIVE_VERIFIED',flush=True);return
    start=prefix.stat().st_size if prefix.exists() else 0
    if start>total:raise ValueError('Oversize prefix')
    if shutil.disk_usage(a.directory).free<2*total+1024**3:raise RuntimeError('Insufficient assembly space')
    pieces=a.directory/(a.archive+'.ranges');pieces.mkdir(exist_ok=True);url=json.loads(a.urls.read_text())[a.archive]
    ranges=[(i,min(i+64*1024**2,total)-1) for i in range(start,total,64*1024**2)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks=[pool.submit(fetch_range,url,pieces/f'{lo}-{hi}.bin',lo,hi,total) for lo,hi in ranges];done=start
        for task in as_completed(tasks):done+=task.result();print(f'RANGE_PROGRESS {done}/{total} {done/total:.1%}',flush=True)
    assembled=target.with_name(target.name+'.assembled')
    with assembled.open('wb') as output:
        if start:
            with prefix.open('rb') as f:shutil.copyfileobj(f,output,8*1024**2)
        for lo,hi in ranges:
            with (pieces/f'{lo}-{hi}.bin').open('rb') as f:shutil.copyfileobj(f,output,8*1024**2)
    if assembled.stat().st_size!=total or checksum(assembled)!=info['sha256']:raise ValueError('Final archive SHA256 mismatch')
    os.replace(assembled,target);print('ARCHIVE_SHA256_VERIFIED '+a.archive,flush=True)

if __name__=='__main__':main()
