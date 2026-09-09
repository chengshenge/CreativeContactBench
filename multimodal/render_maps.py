"""Re-render fixed-scale sensor maps from a solver snapshot (fonts may vary)."""
import argparse,json
from pathlib import Path
import numpy as np
from pressure_projection import project_snapshot
SURFACES=('left_finger_inner','right_finger_inner','left_finger_distal','right_finger_distal')
SHORT_NAMES=('Left inner pad','Right inner pad','Left fingertip end','Right fingertip end')
NORMAL_MAX_KPA=1500.
SHEAR_MAX_KPA=750.
CELL_SIZE_M=.002

def cropped(spec,key):
    array=spec[key]
    edges=spec['v_edges_m']
    if spec['face']=='inner':
        cut=int(np.searchsorted(edges,.032-1e-10))
        for channel in ('normal_force_N','shear_u_force_N','shear_v_force_N'):
            if np.max(np.abs(spec[channel][:cut]),initial=0)>1e-10:
                raise RuntimeError('Fixed display ROI would hide nonzero force')
        array=array[cut:]
        edges=edges[cut:]
    extent=[spec['u_edges_m'][0]*1000,spec['u_edges_m'][-1]*1000,edges[0]*1000,edges[-1]*1000]
    return array,extent

def render_map(result,path,normal_only=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':9})
    normal_norm=Normalize(0,NORMAL_MAX_KPA)
    shear_norm=Normalize(-SHEAR_MAX_KPA,SHEAR_MAX_KPA)
    if normal_only:
        fig,axes=plt.subplots(2,2,figsize=(6,6.3),layout='constrained')
        for ax,name,title in zip(axes.flat,SURFACES,SHORT_NAMES):
            spec=result['surfaces'][name]
            a,extent=cropped(spec,'pressure_kPa')
            im=ax.imshow(a,extent=extent,origin='lower',interpolation='nearest',cmap='viridis',norm=normal_norm,aspect='equal')
            ax.set_title(title)
            ax.set_xlabel('u (mm)');ax.set_ylabel('v (mm)')
            ax.set_xticks([-10,0,10]);ax.set_yticks([34,44,54] if spec['face']=='inner' else [0,14,28])
        fig.colorbar(im,ax=axes,label='Simulated normal pressure (kPa)',shrink=.85,ticks=[0,500,1000,1500])
        fig.suptitle('Simulated normal-pressure maps\nFixed 2 mm grid and shared scale',fontsize=12)
    else:
        fig,axes=plt.subplots(4,3,figsize=(9,9.6),layout='constrained')
        channels=[('pressure_kPa','normal_force_N','Normal pressure p','viridis',normal_norm),
                  ('shear_u_kPa','shear_u_force_N','Signed shear tau_u','RdBu_r',shear_norm),
                  ('shear_v_kPa','shear_v_force_N','Signed shear tau_v','RdBu_r',shear_norm)]
        for row,(name,title) in enumerate(zip(SURFACES,SHORT_NAMES)):
            spec=result['surfaces'][name]
            for col,(key,force_key,channel,cmap,norm) in enumerate(channels):
                ax=axes[row,col]
                a,extent=cropped(spec,key)
                im=ax.imshow(a,extent=extent,origin='lower',interpolation='nearest',cmap=cmap,norm=norm,aspect='equal')
                ax.set_title(channel if row==0 else '')
                ax.set_ylabel(title+'\nv (mm)' if col==0 else 'v (mm)')
                ax.set_xlabel(f'u (mm)   F = {float(spec[force_key].sum()):+.3f} N')
                ax.set_xticks([-10,0,10]);ax.set_yticks([34,44,54] if spec['face']=='inner' else [0,14,28])
        from matplotlib.cm import ScalarMappable
        fig.colorbar(ScalarMappable(norm=normal_norm,cmap='viridis'),ax=axes[:,0],location='bottom',
                     label='Normal pressure (kPa)',ticks=[0,500,1000,1500],shrink=.92,pad=.035)
        fig.colorbar(ScalarMappable(norm=shear_norm,cmap='RdBu_r'),ax=axes[:,1:],location='bottom',
                     label='Signed tangential force density (kPa)',ticks=[-750,-375,0,375,750],shrink=.88,pad=.035)
        fig.suptitle('Simulated pressure and shear maps\n2 mm grid; fixed scales; F = area-integrated force',fontsize=13)
    fig.savefig(path,dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--normal-only",action="store_true")
    args=parser.parse_args()
    result=project_snapshot(json.loads(args.snapshot.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    render_map(result,args.output,args.normal_only)
