import tempfile
from pathlib import Path
from unittest import TestCase,main
from unittest.mock import patch
from test_download import Response
from download_ranges import fetch_range
class Tests(TestCase):
    def test_bounded_range(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'chunk'
            with patch('urllib.request.urlopen',return_value=Response(b'def',206,{'Content-Range':'bytes 3-5/9'})) as request:
                self.assertEqual(fetch_range('https://example.org/x',p,3,5,9),3)
                self.assertEqual(request.call_args.args[0].headers['Range'],'bytes=3-5')
            self.assertEqual(p.read_bytes(),b'def')
    def test_whole_file_response_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'chunk'
            with patch('urllib.request.urlopen',return_value=Response(b'def',200)):
                with self.assertRaises(ValueError):fetch_range('https://example.org/x',p,3,5,9)
            self.assertFalse(p.exists())
if __name__=='__main__':main()
