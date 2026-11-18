"""Checks of the existing-Java orchestration, not speed/quality evidence."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from ml import shape_java_comparison as workflow


class JavaProfileGuards(unittest.TestCase):
    def config(self):
        return dict(java="java",java_classpath="classes:math.jar",heap="1g",histogram_dir="histograms",
                    merge_rate=.5,epsilon=.25,delta=.05,num_files=4)

    def test_four_java_predictor_modes_use_existing_engine(self):
        cfg=self.config()
        for method in workflow.METHODS:
            argv=workflow._java_args(cfg,Path("dev"),method,0,"q.csv","out.csv","setup.csv")
            self.assertIn(workflow.MAIN_CLASS,argv)
            self.assertEqual(argv[argv.index("-mode")+1],"approx")
            self.assertEqual("-modelPath" in argv,method in ("ML_v1","ML_v1_shape"))
            self.assertEqual("-shapeDir" in argv,method in ("R10","ML_v1_shape"))
            self.assertEqual("-augmentShape" in argv,method=="ML_v1_shape")

    def test_exact_mode_does_not_load_model_or_synopsis(self):
        argv=workflow._java_args(self.config(),Path("dev"),"always_exact",0,"q.csv","o.csv","s.csv",
                                 mode="exact",raw_data_dir="raw")
        self.assertNotIn("-runDir",argv)
        self.assertNotIn("-modelPath",argv)
        self.assertEqual(argv[argv.index("-mode")+1],"exact")

    def test_query_writer_refuses_truth_and_preserves_only_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"q.csv"
            frame=pd.DataFrame(dict(qId=[97],x0=[0.],x1=[1.],y0=[2.],y1=[3.]))
            workflow._write_queries(path,frame)
            self.assertEqual(path.read_text().strip(),"0.0,1.0,2.0,3.0")
            with self.assertRaisesRegex(ValueError,"omit offline labels"):
                workflow._write_queries(path,frame.assign(truth=100))

    def test_java_row_ids_are_remapped_without_a_truth_lookup(self):
        geometry=pd.DataFrame(dict(qId=[41,99],x0=[0.,1.],x1=[1.,2.],y0=[0.,0.],y1=[1.,1.]))
        def fake_run(argv,**kwargs):
            path=Path(argv[argv.index("-outFile")+1])
            pd.DataFrame(dict(qId=[0,1],mergeRate=[.5,.5],answer=[10.,11.],fromSynopsis=[False,False],
                estRelError=[.1,.2],errorScore=[.1,.2],cutoff=[0.,0.],gateNs=[10,10],synNs=[5,5],
                rawNs=[0,0],exact=[np.nan,np.nan],relErr=[np.nan,np.nan])).to_csv(path,index=False)
            return SimpleNamespace(returncode=0,stdout="ok",stderr="")
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(workflow.subprocess,"run",side_effect=fake_run):
            frame,seconds=workflow._run_java(self.config(),Path("dev"),"ML_v1",0,geometry,Path(tmp)/"one")
            self.assertEqual(frame.qId.tolist(),[41,99])
            self.assertGreaterEqual(seconds,0)
            record=json.loads((Path(tmp)/"one/command.json").read_text())
            self.assertFalse(record["query_file_has_truth"])

    def test_changed_java_code_or_data_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/"class"
            file.write_bytes(b"original")
            identity=dict(files_sha256={str(file):workflow._sha(file)})
            workflow._check_files(identity)
            file.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError,"identity changed"):
                workflow._check_files(identity)


if __name__=="__main__":unittest.main()
