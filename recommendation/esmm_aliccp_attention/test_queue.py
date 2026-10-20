"""Fail closed if the baseline is running, incomplete, or incompatible."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_when_ready as runner


class QueueTests(unittest.TestCase):
    def fixture(self, root):
        reference = root/'baseline'
        reference.mkdir()
        (reference/'summary.json').write_text('{"best_epoch":1}')
        (reference/'checkpoint_fixed.json').write_text('{}')
        (reference/'config.json').write_text('{"lr":0.0001}')
        reference.with_suffix('.log').write_text('TRAINING_COMPLETE\n')
        output = root/'attention'
        args = ['run_when_ready.py','--reference-run',str(reference),'--output',str(output),
                '--data-dir',str(root/'data'),'--wait-hours','0']
        return reference, output, args

    def test_success_launches_expected_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            reference, output, args = self.fixture(Path(temporary))
            class Child:
                pid = 123
                def __enter__(self):return self
                def __exit__(self, *unused):pass
                def wait(self):
                    output.mkdir()
                    (output/'summary.json').write_text('{}')
                    return 0
            with patch('sys.argv',args),patch.object(runner,'matching_train_processes',return_value=[]), \
                 patch.object(runner.subprocess,'run',return_value=SimpleNamespace(stdout='1\n')), \
                 patch.object(runner.subprocess,'Popen',return_value=Child()) as launch, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(),0)
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index('--lr')+1],'0.0001')
            self.assertEqual(command[command.index('--reference-run')+1],str(reference.resolve()))
            self.assertEqual(command[command.index('--attention-heads')+1],'4')
            self.assertEqual(json.loads(output.with_suffix('.queue.json').read_text())['status'],'complete')

    def test_incomplete_or_incompatible_baseline_never_launches(self):
        for invalid in ['missing_marker','different_lr']:
            with tempfile.TemporaryDirectory() as temporary:
                reference, output, args = self.fixture(Path(temporary))
                if invalid=='missing_marker':reference.with_suffix('.log').write_text('FAILED\n')
                else:(reference/'config.json').write_text('{"lr":0.0003}')
                with patch('sys.argv',args),patch.object(runner,'matching_train_processes',return_value=[]), \
                     patch.object(runner.subprocess,'Popen') as launch, \
                     contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(RuntimeError):runner.main()
                launch.assert_not_called()
                self.assertEqual(json.loads(output.with_suffix('.queue.json').read_text())['status'],'failed')

    def test_running_baseline_prevents_launch_even_with_old_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            reference, output, args = self.fixture(Path(temporary))
            with patch('sys.argv',args),patch.object(runner,'matching_train_processes',return_value=[123]), \
                 patch.object(runner.subprocess,'Popen') as launch, \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'waiting for baseline'):runner.main()
            launch.assert_not_called()


if __name__=='__main__':unittest.main()
