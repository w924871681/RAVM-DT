import unittest
from joint_gate_reference import joint_decision

class TestJointGate(unittest.TestCase):
    def g(self,d=-.0003,p=.5,w=.15,m=.0001,parent=.003):
        return joint_decision(d,parent,p,margin=m,weight=w)
    def test_recentmean_nested(self):
        for d in [-.01,-.0001,-.000001,0,.00001]:
            for p in [0,.2,.8,1]:
                a=self.g(d,p,w=0,m=.00005).commit
                b=(d<=-.00005)
                self.assertEqual(a,b)
    def test_pivot_equals_mean(self):
        self.assertAlmostEqual(self.g(-.0002,.2).score,-.0002)
    def test_monotonic_mwu_p(self):
        arr=[self.g(-.0002,p).score for p in (0,.2,.5,1)]
        self.assertEqual(arr,sorted(arr))
    def test_strong_distribution_can_rescue_small_mean(self):
        self.assertFalse(self.g(-.00008,.2,m=.0001).commit)
        self.assertTrue(self.g(-.00008,0,m=.0001).commit)
    def test_weak_distribution_can_block(self):
        self.assertTrue(self.g(-.00025,.2,m=.0001).commit)
        self.assertFalse(self.g(-.00025,1,w=.3,m=.0001).commit)
    def test_positive_recent_mean_cannot_pass(self):
        self.assertFalse(self.g(.00001,0,w=10,m=0).commit)
    def test_not_equivalent_to_hard_conjunction(self):
        # p=.35 would fail MWU_MEAN hard p<.2, but has enough mean gain here.
        self.assertTrue(self.g(-.001,.35,w=.1,m=.0001).commit)
    def test_input_errors(self):
        for val in [float('nan'),float('inf')]:
            with self.assertRaises(ValueError):self.g(d=val)
        with self.assertRaises(ValueError):self.g(p=-.1)
        with self.assertRaises(ValueError):self.g(p=1.1)
        with self.assertRaises(ValueError):self.g(parent=-1)
        with self.assertRaises(ValueError):self.g(w=-1)
        with self.assertRaises(ValueError):self.g(m=-1)

if __name__=='__main__':unittest.main()
