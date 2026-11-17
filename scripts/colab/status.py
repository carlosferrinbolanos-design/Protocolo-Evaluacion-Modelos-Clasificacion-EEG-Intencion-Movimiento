import subprocess
print(subprocess.run('tail -n 4 /content/w/pipeline.log; tail -n 2 /content/w/dl.log /content/w/extract.log /content/w/cl.log /content/w/deep.log 2>/dev/null; ls /content/raw/2017 2>/dev/null | grep -c gdf$; du -sh /content/raw 2>/dev/null; ls /content/out | wc -l; free -g | sed -n 2p',shell=True,capture_output=True,text=True).stdout)
