import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from download import download,extract_raw,validate_gzip


class Response(io.BytesIO):
    def __init__(self,body,status=200,headers=None):
        super().__init__(body);self.status=status;self.headers=headers or {}


class Tests(unittest.TestCase):
    def test_resume_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'x.gz';target.with_name('x.gz.part').write_bytes(b'abc')
            with patch('urllib.request.urlopen',return_value=Response(b'def',206,{'Content-Range':'bytes 3-5/6'})) as request:
                download('https://example.org/x',target,6)
                self.assertEqual(request.call_args.args[0].headers['Range'],'bytes=3-')
            self.assertEqual(target.read_bytes(),b'abcdef')

    def test_ignored_range_restarts_cleanly(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'x.gz';target.with_name('x.gz.part').write_bytes(b'abc')
            with patch('urllib.request.urlopen',return_value=Response(b'abcdef')):download('https://example.org/x',target,6)
            self.assertEqual(target.read_bytes(),b'abcdef')

    def test_bad_range_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'x.gz';target.with_name('x.gz.part').write_bytes(b'abc')
            with patch('urllib.request.urlopen',return_value=Response(b'def',206,{'Content-Range':'bytes 0-2/6'})):
                with self.assertRaises(ValueError):download('https://example.org/x',target,6)
            self.assertFalse(target.exists())

    def test_html_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'x.gz'
            with patch('urllib.request.urlopen',return_value=Response(b'abc',headers={'Content-Type':'text/html'})):
                with self.assertRaises(ValueError):download('https://example.org/x',target,3)
            self.assertFalse(target.exists())

    @patch('download.shutil.disk_usage')
    def test_extract_checks_members_and_round_trip(self,disk_usage):
        disk_usage.return_value.free=100*1024**3
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'train.gz';output=Path(folder)/'raw'
            with tarfile.open(path,'w:gz') as archive:
                for name in ['common_features_train.csv','sample_skeleton_train.csv']:
                    info=tarfile.TarInfo('sample_train/'+name);info.size=4
                    archive.addfile(info,io.BytesIO(b'row\n'))
            self.assertGreater(validate_gzip(path),0)
            report=extract_raw(path,output,'train')
            self.assertEqual(len(report['files']),2)
            self.assertEqual(report,extract_raw(path,output,'train'))
            (output/'common_features_train.csv').write_text('changed')
            with self.assertRaises(ValueError):extract_raw(path,output,'train')

    def test_archive_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'train.gz'
            with tarfile.open(path,'w:gz') as archive:
                info=tarfile.TarInfo('sample_skeleton_train.csv');info.type=tarfile.SYMTYPE;info.linkname='/etc/passwd';archive.addfile(info)
            with self.assertRaises(ValueError):extract_raw(path,Path(folder)/'raw','train')


if __name__=='__main__':unittest.main()
