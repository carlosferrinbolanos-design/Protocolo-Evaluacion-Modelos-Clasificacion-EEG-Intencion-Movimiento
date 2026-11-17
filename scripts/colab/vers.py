import subprocess,sys
print(sys.version.split()[0])
print(subprocess.run([sys.executable,'-c','import numpy,scipy,sklearn,mne,pyriemann,torch,pandas,matplotlib;print(numpy.__version__,scipy.__version__,sklearn.__version__,mne.__version__,pyriemann.__version__,torch.__version__,pandas.__version__,matplotlib.__version__)'],capture_output=True,text=True).stdout)
print(subprocess.run('tail -n 3 /content/w/pipeline.log; tail -n 2 /content/w/deep.log; ls /content/out | wc -l',shell=True,capture_output=True,text=True).stdout)
