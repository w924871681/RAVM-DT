"""Read sealed CSVs; render descriptive figures only. No fitting or data mutation."""
from pathlib import Path
import csv, json, hashlib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

R=Path(__file__).resolve().parents[1]
D=R/'05_Scientific_Traceability/data'
F=R/'01_Clean/figures' if (R/'01_Clean').exists() else R.parent/'READER_OUTPUT/Figures'
F.mkdir(parents=True,exist_ok=True)
J='JOINT__l3__m0'; DU='DUAL_MEAN__g12'; ETA=5.6036475532487204e-5
ORDER=[J,DU,'RECENT_MEAN__g12','MWU_MEAN__g04','RAVM_D2B_V02','RAVM_ORIGINAL','ALWAYS_FT','ER','EWC','HUBER_FT','VALIDATION_COMMIT','FROZEN']
N={J:'Joint Gate',DU:'DualMean','RECENT_MEAN__g12':'RecentMean','MWU_MEAN__g04':'MWU-Mean','RAVM_D2B_V02':'RAVM-D2B','RAVM_ORIGINAL':'Original RAVM','ALWAYS_FT':'Always-FT','ER':'ER','EWC':'EWC','HUBER_FT':'Huber-FT','VALIDATION_COMMIT':'ValidationCommit','FROZEN':'Frozen'}
COL=['#7297B8','#DDA85B','#89B5A7','#AA96B5','#B7A16C','#9199A3','#C8C5C0','#8CABB8','#B5877E','#96AC88','#A7A7A7','#C9C7C4']
C=dict(zip(ORDER,COL)); manifest=[]; values={}; layout=[]
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.serif':['Times New Roman','DejaVu Serif'],'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#444444','axes.linewidth':.65,'grid.color':'#d7d7d7','grid.linewidth':.45,'grid.linestyle':':','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','hatch.linewidth':.5,'savefig.dpi':400})
def read(n):
 p=D/n; b=p.read_bytes(); rows=list(csv.DictReader(b.decode('utf-8-sig').splitlines()))
 manifest.append({'file':n,'sha256':hashlib.sha256(b).hexdigest(),'rows':len(rows)})
 return rows
def axes_style(ax,axis='x'):
 ax.set_axisbelow(True); ax.grid(axis=axis); ax.tick_params(length=2.5,width=.6)
def save(fig,name,record):
 for ax in fig.axes:
  lo,hi=sorted(ax.get_xlim()); ax.set_xticks([t for t in ax.get_xticks()if lo-1e-9<=t<=hi+1e-9])
  lo,hi=sorted(ax.get_ylim()); ax.set_yticks([t for t in ax.get_yticks()if lo-1e-9<=t<=hi+1e-9])
 for text in fig.findobj(matplotlib.text.Text):
  if text.get_visible() and text.get_text():
   text.set_fontsize(max(8.0,text.get_fontsize()));text.set_fontfamily('Arial');text.set_color('black')
 fig.canvas.draw()
 # A deterministic text-boundary audit complements visual review.
 renderer=fig.canvas.get_renderer(); bounds=fig.bbox; outside=[]
 for t in fig.findobj(matplotlib.text.Text):
  if not t.get_visible()or not t.get_text():continue
  box=t.get_window_extent(renderer)
  if box.width and (box.x0<bounds.x0-2 or box.y0<bounds.y0-2 or box.x1>bounds.x1+2 or box.y1>bounds.y1+2):outside.append(t.get_text())
 layout.append({'figure':name,'text_outside_canvas':outside,'min_font_pt':min(t.get_fontsize()for t in fig.findobj(matplotlib.text.Text)if t.get_visible()and t.get_text())})
 assert not outside,(name,outside)
 for ext in ['pdf','svg','png']:fig.savefig(F/(name+'.'+ext),dpi=400,facecolor='white')
 fig.savefig(F/(name+'_gray.png'),dpi=150,facecolor='white')
 # Gray preview is a review artifact, not a different publication chart.
 from PIL import Image,ImageOps
 p=F/(name+'_gray.png');ImageOps.grayscale(Image.open(p)).save(p)
 values[name]=record;plt.close(fig)
def fourbars(rows,name,metrics,grouped=False):
 fig,axs=plt.subplots(2,2,figsize=(6.4,4.0 if not grouped else 4.4));fig.subplots_adjust(left=.16,right=.975,bottom=.11,top=.95,wspace=.43,hspace=.43)
 y=np.arange(12)
 for ax,(key,scale,title,xlabel)in zip(axs.flat,metrics):
  if grouped:
   for s,offset,color,hatch in [('clean',-.16,'#DCE6EE','//'),('mixed',.16,'#7297B8','')]:
    xx=np.array([float(next(r for r in rows if r['arm']==a and r['stream']==s)[key])*scale for a in ORDER]);ax.barh(y+offset,xx,height=.29,color=color,edgecolor='#647788',lw=.55,hatch=hatch,label=s.capitalize())
   ax.set_xlim(0, max(float(r[key])*scale for r in rows)*1.12 if key!='commit_rate'else 108)
  else:
   xx=[float(next(r for r in rows if r['arm']==a)[key])*scale for a in ORDER]
   ax.barh(y,xx,height=.66,color=[C[a]for a in ORDER],edgecolor='#656565',linewidth=.45)
   ax.patches[0].set_hatch('///')
   ax.set_xlim(0,max(xx)*1.20 if key!='balanced_commit'else 115)
   for yi,x in zip(y,xx):ax.text(x+ax.get_xlim()[1]*.012,yi,f'{x:.2f}'if key!='balanced_commit'else f'{x:.1f}',va='center',fontsize=8)
  ax.set_yticks(y,[N[a]for a in ORDER]);ax.invert_yaxis();ax.get_yticklabels()[0].set_fontweight('bold');ax.set_xlabel(xlabel);ax.set_title(title,pad=7);axes_style(ax)
  if 'harm' in key:ax.axvline(5,color='#777777',ls='--',lw=.7,zorder=0)
  if 'downside'in key:ax.axvline(ETA*scale,color='#777777',ls='--',lw=.7,zorder=0)
 if grouped:fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.56,1))
 save(fig,name,rows)

main=read('E0_MAIN_TABLES/RESULTS_BY_ARM.csv')
fourbars(main,'fig3_overall_baseline',[
 ('balanced_mse',1e3,'(a) Balanced lifecycle MSE',r'MSE ($\times10^{-3}$)'),
 ('worst_harmful',100,'(b) Worst-stream harmful commitments','Harm / all opportunities (%)'),
 ('worst_downside',1e5,'(c) Worst-stream average downside',r'Downside ($\times10^{-5}$)'),
 ('balanced_commit',100,'(d) Balanced commitment','Commit / all opportunities (%)')])

e1=[r for r in read('E1/METHOD_STREAM.csv')if r['arm']==J]
fig,axs=plt.subplots(2,2,figsize=(6.4,3.3));fig.subplots_adjust(left=.11,right=.98,bottom=.12,top=.88,hspace=.66,wspace=.30)
for ax,(key,scale,title,label)in zip(axs.flat,[('mse',1e3,'(a) Lifecycle MSE',r'MSE ($\times10^{-3}$)'),('harm_frequency',100,'(b) Harmful commitments','Harm / opportunities (%)'),('mean_downside',1e5,'(c) Average downside',r'Downside ($\times10^{-5}$)'),('commit_rate',100,'(d) Commitment','Commit / opportunities (%)')]):
 for kind,labelkind,col,mark,ls in [('mixed','Mixed','#7297B8','o','-'),('persistent_bias','Persistent bias','#DDA85B','s','--'),('replay','Replay','#9199A3','^',':')]:
  rr=[next(r for r in e1 if r['stream']==kind+'_'+s)for s in ['a005','a010','a020']]
  ax.plot([.05,.1,.2],[float(r[key])*scale for r in rr],color=col,marker=mark,ls=ls,ms=4,lw=1.25,label=labelkind)
 ax.set_title(title,pad=7);ax.set_xticks([.05,.1,.2],['0.05','0.10','0.20']);ax.set_xlabel(r'Registered corruption fraction $\alpha$');ax.set_ylabel(label);ax.set_ylim(bottom=0);axes_style(ax,'y')
 if key=='harm_frequency':ax.axhline(5,color='#777777',ls='--',lw=.7);ax.text(.055,5.3,'Source criterion',fontsize=8,color='#555555')
 if key=='mean_downside':ax.axhline(ETA*scale,color='#777777',ls='--',lw=.7)
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.995));save(fig,'fig4_corruption_severity',e1)

benign=read('E0_MAIN_TABLES/BENIGN_REGIME_ACTIVITY_RECALL.csv');regimes=['stationary','gradual','abrupt','recurring'];y=np.arange(4)
fig,axs=plt.subplots(1,2,figsize=(6.4,2.0));fig.subplots_adjust(left=.115,right=.98,bottom=.20,top=.78,wspace=.40)
for a,off,marker in [('ALWAYS_FT',-.16,'s'),(J,0,'o'),('FROZEN',.16,'X')]:
 rr=[next(r for r in benign if r['arm']==a and r['stream']=='clean'and r['regime']==g)for g in regimes]
 axs[0].scatter([float(r['mse'])*1e3 for r in rr],y+off,s=23,facecolors='white'if a!='FROZEN'else C[a],edgecolors=C[a],marker=marker,label=N[a],zorder=3)
for key,off,col,mark,label in [('commit_rate',-.08,'#7297B8','o','Commitment'),('beneficial_commit_recall',.08,'#DDA85B','s','Beneficial-commit recall')]:
 rr=[next(r for r in benign if r['arm']==J and r['stream']=='clean'and r['regime']==g)for g in regimes]
 axs[1].scatter([float(r[key])*100 for r in rr],y+off,s=25,facecolors='white',edgecolors=col,marker=mark,label=label,zorder=3)
for ax,title,xlabel in zip(axs,['(a) Clean lifecycle MSE','(b) Joint Gate activity / recall'],[r'MSE ($\times10^{-3}$)','Rate (%)']):
 ax.set_yticks(y,[g.capitalize()for g in regimes]);ax.invert_yaxis();ax.set_xlabel(xlabel);ax.set_title(title,pad=8);axes_style(ax);ax.legend(loc='lower center',bbox_to_anchor=(.5,1.16),ncol=2 if ax==axs[1]else 3,frameon=False,columnspacing=.8,handletextpad=.25,fontsize=8)
save(fig,'fig5_benign_evolution',[r for r in benign if r['stream']=='clean'])

paired=read('E0_MAIN_TABLES/PAIRED_MSE_ALL_BASELINES.csv');disc=read('E0_MAIN_TABLES/SAME_STATE_DISAGREEMENT_SUMMARY.csv')[:3]
fig,axs=plt.subplots(1,2,figsize=(6.4,2.8),gridspec_kw={'width_ratios':[1.12,1]});fig.subplots_adjust(left=.16,right=.98,bottom=.18,top=.84,wspace=.58)
ax=axs[0];yy=np.arange(len(paired));xx=np.array([float(r['mse_delta'])*1e3 for r in paired]);lo=np.array([float(r['lower95'])*1e3 for r in paired]);hi=np.array([float(r['upper95'])*1e3 for r in paired])
ax.errorbar(xx,yy,xerr=[xx-lo,hi-xx],fmt='o',ms=3.4,color='#7297B8',capsize=2,lw=.9);ax.axvline(0,color='#777777',ls='--',lw=.7);ax.set_yticks(yy,[N[r['baseline']]for r in paired]);ax.invert_yaxis();ax.set_xlabel(r'Joint $-$ comparator MSE ($\times10^{-3}$)');ax.set_title('(a) All eleven utility contrasts',pad=10);axes_style(ax)
ax=axs[1];base=np.zeros(3)
for k,col,hatch,label in [('a_only_commit','#7297B8','//','Joint only'),('b_only_commit','#DDA85B','','Control only')]:
 v=np.array([int(r[k])/6912*100 for r in disc]);ax.barh(np.arange(3),v,left=base,height=.5,color=col,edgecolor='#777777',lw=.5,hatch=hatch,label=label);base+=v
for i,r in enumerate(disc):ax.text(base[i]+.25,i,str(r['different'])+'/6,912',va='center',fontsize=8)
ax.set_yticks(range(3),['DualMean','MWU-Mean','RecentMean']);ax.invert_yaxis();ax.set_xlim(0,22);ax.set_xlabel('Disagreeing actions (%)');ax.set_title('(b) Identical-state gate decisions',pad=10);axes_style(ax);ax.legend(loc='lower center',bbox_to_anchor=(.5,1.10),frameon=False,ncol=2,fontsize=8)
save(fig,'fig6_mechanism_analysis',{'paired':paired,'same_state':disc})

dev=read('E0_MAIN_TABLES/DEVCAL_JOINT_RESULTS_BY_ARM.csv');weights=[0,.05,.15,.3];mult=[1,5,10,20]
fig=plt.figure(figsize=(6.4,3.85));gs=fig.add_gridspec(2,2,height_ratios=[1,1.15],left=.11,right=.97,bottom=.105,top=.92,hspace=.72,wspace=.40)
maps=[LinearSegmentedColormap.from_list('muted_blue',['#F5F8FA','#B8CBDC','#6089AF']),LinearSegmentedColormap.from_list('muted_orange',['#FFF9EE','#EDCB93','#C48540'])]
for k,(field,scale,title)in enumerate([('balanced_mse',1e3,'(a) DevCal balanced MSE'),('worst_harmful',100,'(b) DevCal worst-stream harm (%)')]):
 ax=fig.add_subplot(gs[0,k]);arr=np.array([[float(next(r for r in dev if float(r['weight'])==w and abs(float(r['threshold'])-m*ETA)<1e-12)[field])*scale for m in mult]for w in weights]);ax.imshow(arr,cmap=maps[k],aspect='auto')
 for i,w in enumerate(weights):
  for j,m in enumerate(mult):
   rr=next(r for r in dev if float(r['weight'])==w and abs(float(r['threshold'])-m*ETA)<1e-12);q='Q'if rr['dev_qualified']=='True'else 'x';ax.text(j,i,f'{arr[i,j]:.3f}\n{q}'if k==0 else f'{arr[i,j]:.1f}\n{q}',ha='center',va='center',fontsize=8,color='#152B3D')
 ax.add_patch(Rectangle((-.48,2.52),.96,.96,fill=False,edgecolor='#1F3B52',lw=2));ax.set_xticks(range(4),mult);ax.set_yticks(range(4),weights);ax.set_xlabel(r'Margin multiplier $\delta/\eta$');ax.set_ylabel(r'Weight $\lambda$');ax.set_title(title,pad=9);ax.tick_params(length=0)
ax=fig.add_subplot(gs[1,:]);e2=[]
for arch in ['GRU','MLP','TCN']:
 for h in [1,4]:
  group=f'E2_{arch}_H{h}';pp=read(group+'/PAIRED_ALL_BASELINES.csv');rr=next(r for r in pp if r['baseline']==DU and r['metric']=='mse'and r['stream']=='BALANCED_CLEAN_MIXED');risk=next(r for r in read(group+'/METHOD_STREAM.csv')if r['arm']==J and r['stream']=='mixed');e2.append((group,rr,risk))
for i,(g,r,risk)in enumerate(e2):
 x=float(r['delta_joint_minus_baseline'])*1e3;l=float(r['ci_low'])*1e3;h=float(r['ci_high'])*1e3;col='#7297B8'if g.endswith('H1')else '#DDA85B';ax.errorbar(x,i,xerr=[[x-l],[h-x]],fmt='o'if g.endswith('H1')else 's',ms=4,capsize=2,color=col,lw=1)
ax.set_yticks(range(6),[g[3:].replace('_',' / ')for g,_,_ in e2]);ax.invert_yaxis();ax.axvline(0,color='#777777',ls='--',lw=.7);ax.set_xlabel(r'Joint $-$ DualMean balanced MSE ($\times10^{-3}$)');ax.set_title('(c) Architecture / horizon transfer: all H4 mixed streams fail risk criteria',fontsize=8.2,pad=8);axes_style(ax)
save(fig,'fig7_sensitivity_generality',{'devcal':dev,'generality':e2})

e3=read('E3/METHOD_STREAM.csv')
fourbars(e3,'fig8_alibaba_external',[
 ('mse',1e3,'(a) Lifecycle MSE',r'MSE ($\times10^{-3}$)'),('harm_frequency',100,'(b) Harmful commitments','Harm / all opportunities (%)'),('mean_downside',1e5,'(c) Average downside',r'Downside ($\times10^{-5}$)'),('commit_rate',100,'(d) Commitment','Commit / all opportunities (%)')],True)
REPORT=R/'07_Figure_Source' if (R/'01_Clean').exists() else F
(REPORT/'PLOTTED_VALUES.json').write_text(json.dumps(values,indent=2),'utf-8')
(REPORT/'FIGURE_INPUT_HASHES.json').write_text(json.dumps(manifest,indent=2),'utf-8')
(REPORT/'LAYOUT_CHECK.json').write_text(json.dumps(layout,indent=2),'utf-8')
print(json.dumps({'figures':len(values),'input_reads':len(manifest),'layout':layout},indent=2))
