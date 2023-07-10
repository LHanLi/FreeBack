

def matplot(r=1, c=1, sharex=False, sharey=False, w=8, d=5):
  import seaborn as sns
  import matplotlib.pyplot as plt
  # don't use sns style
  sns.reset_orig()
  #plot
  #run configuration 
  plt.rcParams['font.size']=14
  plt.rcParams['font.family'] = 'KaiTi'
  #plt.rcParams['font.family'] = 'Arial'
  plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
  plt.rcParams['axes.linewidth']=1
  plt.rcParams['axes.grid']=True
  plt.rcParams['grid.linestyle']='--'
  plt.rcParams['grid.linewidth']=0.2
  plt.rcParams["savefig.transparent"]='True'
  plt.rcParams['lines.linewidth']=0.8
  plt.rcParams['lines.markersize'] = 1
  
  #保证图片完全展示
  plt.tight_layout()
    
  #subplot
  fig,ax = plt.subplots(r,c,sharex=sharex, sharey=sharey,figsize=(w,d))
  plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace = None, wspace=0.5)
  
  return plt, fig, ax