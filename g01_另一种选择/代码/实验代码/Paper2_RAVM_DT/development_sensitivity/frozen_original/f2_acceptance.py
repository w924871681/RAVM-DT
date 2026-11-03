"""Run supplied mathematics QA and execution-boundary acceptance before scientific data."""
from d3_common import *
import unittest
def main():
 protect_originals();verify_science_lock();assert not list((ROOT/'data').glob('center_*.npz'))
 exec(compile((ROOT/'qa_validate.py').read_text(encoding='utf8'),str(ROOT/'qa_validate.py'),'exec'),dict(__file__=str(ROOT/'qa_validate.py'),__name__='qa_validate'))
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT),pattern='test_*.py'));assert result.wasSuccessful()
 atomic(ROOT/'engineering/MATH_AND_EXECUTION_TESTS.json',dict(status='PASS',tests=result.testsRun,optimizer_steps=0,scientific_data_generated=False,at_unix=time.time()))
if __name__=='__main__':main()
