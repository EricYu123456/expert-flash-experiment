import json,os,pathlib,signal,subprocess,sys,time
role,mode,log=sys.argv[1:];signal.signal(signal.SIGTERM,signal.SIG_IGN)
if role=='root':subprocess.Popen([sys.executable,__file__,'middle',mode,log],start_new_session=True)
elif role=='middle':subprocess.Popen([sys.executable,__file__,'leaf',mode,log],start_new_session=True)
blob=bytearray(256<<20) if role=='leaf' and mode=='rss' else bytearray(1<<20)
for i in range(0,len(blob),4096):blob[i]=1
text=pathlib.Path('/proc/self/stat').read_text();v=text[text.rfind(')')+2:].split();entry={'PID':os.getpid(),'starttime_ticks':int(v[19]),'role':role,'session':os.getsid(0)}
fd=os.open(log,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600);os.write(fd,(json.dumps(entry)+'\n').encode());os.close(fd)
if role=='leaf':pathlib.Path(log+'.ready').write_text('ready')
if role=='root' and mode=='orphan':
 while not pathlib.Path(log+'.ready').exists():time.sleep(.01)
 sys.exit(0)
while True:time.sleep(.05)
