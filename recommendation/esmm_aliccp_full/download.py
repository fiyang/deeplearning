"""Download user-authorized Ali-CCP links with resume, then validate and safely extract."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import time
import urllib.error
import urllib.request


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    os.replace(temporary,path)


def checksum(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def download(url,target,expected_bytes,retries=5):
    target=Path(target);partial=target.with_name(target.name+'.part')
    if not url.startswith('https://'):raise ValueError('Only HTTPS download links are accepted')
    if target.exists():
        if target.stat().st_size!=expected_bytes:raise ValueError('Existing complete file has unexpected size')
        return
    for attempt in range(retries):
        start=partial.stat().st_size if partial.exists() else 0
        if start==expected_bytes:os.replace(partial,target);return
        if start>expected_bytes:raise ValueError('Partial file is larger than the published file size')
        headers={'User-Agent':'AliCCP-Research-Downloader/1.0'}
        if start:headers['Range']=f'bytes={start}-'
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=45) as response:
                mode='ab' if start else 'wb'
                if response.status==206:
                    m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                    if not m or int(m[1])!=start or int(m[3])!=expected_bytes:raise ValueError('Invalid range response')
                elif response.status==200:
                    mode='wb';start=0
                else:raise ValueError('Unexpected HTTP response status')
                if 'text/html' in response.headers.get('Content-Type','').lower():raise ValueError('Received a webpage instead of an archive')
                received=start;last=time.monotonic()
                with partial.open(mode) as f:
                    while block:=response.read(4*1024*1024):
                        if received+len(block)>expected_bytes:raise ValueError('Download exceeds published file size')
                        f.write(block);received+=len(block)
                        if time.monotonic()-last>=15:
                            print(f'DOWNLOAD {target.name} {received}/{expected_bytes} {received/expected_bytes:.1%}',flush=True);last=time.monotonic()
            if received==expected_bytes:os.replace(partial,target);return
            print(f'RESUME {target.name} received={received}',flush=True)
        except urllib.error.HTTPError as exc:
            # Do not print signed URLs or attempt to circumvent access controls.
            if exc.code in [401,403]:raise RuntimeError(f'{target.name}: download permission or link expired (HTTP {exc.code})') from None
            print(f'RETRY {target.name} HTTP {exc.code} attempt={attempt+1}',flush=True)
        except (urllib.error.URLError,TimeoutError,OSError) as exc:
            print(f'RETRY {target.name} {type(exc).__name__} attempt={attempt+1}',flush=True)
        if attempt+1<retries:time.sleep(min(2**attempt,15))
    raise RuntimeError(f'{target.name}: incomplete after {retries} attempts; partial file retained for resume')


def validate_gzip(path):
    # Read through the gzip trailer to validate CRC and uncompressed length.
    total=0
    with gzip.open(path,'rb') as f:
        while block:=f.read(8*1024*1024):total+=len(block)
    return total


def extract_raw(path,output,split):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    marker=output/'extraction.json'
    archive_sha=checksum(path)
    if marker.exists():
        old=json.loads(marker.read_text())
        if old['archive_sha256']!=archive_sha:raise ValueError('Archive changed since extraction')
        for item in old['files']:
            file=output/item['name']
            if not file.exists() or checksum(file)!=item['sha256']:raise ValueError('Extracted file failed checksum')
        return old
    expected={f'common_features_{split}.csv',f'sample_skeleton_{split}.csv'}
    found={};ignored=[];total=0
    with tarfile.open(path,'r|gz') as archive:
        for member in archive:
            if member.isdir():continue
            if not member.isfile():raise ValueError('Archive contains a link or special file')
            name=Path(member.name).name
            if name not in expected:ignored.append(member.name);continue
            if name in found:raise ValueError('Duplicate expected file in archive')
            if member.size<1 or member.size>60*1024**3:raise ValueError('Unexpected raw file size')
            total+=member.size
            if total>80*1024**3:raise ValueError('Archive exceeds conservative extraction size budget')
            if shutil.disk_usage(output).free<member.size+10*1024**3:raise RuntimeError('Insufficient space to extract safely')
            target=output/name;partial=target.with_name(name+'.part');digest=hashlib.sha256();count=0
            source=archive.extractfile(member)
            with source,partial.open('wb') as f:
                while block:=source.read(8*1024*1024):f.write(block);digest.update(block);count+=len(block)
            if count!=member.size:raise ValueError('Truncated archive member')
            os.replace(partial,target)
            found[name]=dict(name=name,bytes=count,sha256=digest.hexdigest(),archive_member=member.name)
            print(f'EXTRACTED {name} bytes={count}',flush=True)
    if set(found)!=expected:raise ValueError('Missing original common-feature or sample-skeleton files')
    result=dict(archive=Path(path).name,archive_sha256=archive_sha,files=list(found.values()),ignored_files=ignored)
    write_json(marker,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--urls',type=Path,help='Private JSON file: archive filename -> authorized HTTPS download link')
    p.add_argument('--data-dir',type=Path,default=Path('../data/aliccp-full'))
    p.add_argument('--manifest',type=Path,default=Path(__file__).with_name('source_manifest.json'))
    p.add_argument('--extract-existing',action='store_true')
    args=p.parse_args();args.data_dir.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.manifest.read_text())
    urls=json.loads(args.urls.read_text()) if args.urls else {}
    if not args.extract_existing and not urls:p.error('--urls is required unless --extract-existing is set')
    if shutil.disk_usage(args.data_dir).free<65*1024**3:raise RuntimeError('At least 65 GiB of free space is required for download and extraction')
    archives=args.data_dir/'archives';archives.mkdir(exist_ok=True);results=[]
    def process(info):
        target=archives/info['name']
        if not args.extract_existing:download(urls[info['name']],target,info['bytes'])
        if target.stat().st_size!=info['bytes']:raise ValueError('Archive does not match the published byte count')
        if info.get('sha256') and checksum(target)!=info['sha256']:raise ValueError('Archive does not match published Git LFS SHA256')
        print(f'VALIDATING_GZIP {target.name}',flush=True)
        uncompressed_bytes=validate_gzip(target)
        split='train' if 'train' in target.name else 'test'
        result=extract_raw(target,args.data_dir/'raw'/split,split)
        result['gzip_uncompressed_bytes']=uncompressed_bytes;result['compressed_bytes']=target.stat().st_size
        return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(process,info) for info in manifest['files']]
        for task in as_completed(tasks):
            results.append(task.result())
            write_json(args.data_dir/'download_manifest.json',dict(source_page=manifest['mirror_page'],archives=results,complete=len(results)==len(manifest['files'])))
    print('FULL_DATA_DOWNLOAD_AND_EXTRACTION_COMPLETE',flush=True)


if __name__=='__main__':main()
