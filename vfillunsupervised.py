import os 
from u_mlinterp import mlinterps,mlinterpe
from u_riointerp import riointerp
from upaths import OUT_TILES_DPATH
import time 

def demvfill_unspervised(outdir,tiles_xdpath,tilename,method='riointerp'):
    #tilenames = os.listdir(tiles_xdpath)
    #for tilename in tilenames:
        #tilename = 'N10E105'
    ti = time.perf_counter()
    dem_ipath = f"{tiles_xdpath}/{tilename}/{tilename}_tdem_DEM__Fw.tif"
    tile_odpath = f"{outdir}/{tilename}/" 
    os.makedirs(tile_odpath, exist_ok=True)
    
    print(os.path.isfile(dem_ipath))
    if method == 'mlinterps':
        print(f'using {method}')
        dem_opath = f"{tile_odpath}/{tilename}_tdem_DEM__mlinterps.tif"
        if not os.path.isfile(dem_opath):
            mlinterps(dem_ipath, dem_opath, model_type='catboost')
        else:
            print(f'file aready created\nsaved at {dem_opath}')

    elif method == 'mlinterpe':
        print(f'using {method}')
        dem_opath = f"{tile_odpath}/{tilename}_tdem_DEM__mlinterpe.tif"
        if not os.path.isfile(dem_opath):
            mlinterpe(dem_ipath, dem_opath, model_type='catboost')
        else:
            print(f'file aready created\nsaved at {dem_opath}')

    elif method == 'riointerp':
        print(f'using {method}')
        dem_opath = f"{tile_odpath}/{tilename}_tdem_DEM__riointerp.tif"
        if not os.path.isfile(dem_opath):
            riointerp(dem_ipath, dem_opath, smoothing_iterations=si)
        else:
            print(f'file aready created\nsaved at {dem_opath}')

    else:
        print('method not available: try mlinterps,mlinterpe,riointerp')
    tf = time.perf_counter() -ti 
    print(f'run.time ={tf/60} min(s)')
    print(f'tilename: {tilename} @{method}')


X = 30
method = 'mlinterpe' #riointerp mlinterps #mlinterpe
si = 0 # smoothing_iterations for riointerp
if __name__ == '__main__':
    ti = time.perf_counter()

    outdir = f"{OUT_TILES_DPATH}/DEMVFILL/TILES{X}"
    tiles_xdpath = f"{OUT_TILES_DPATH}/TILES{X}"
    tilenames = os.listdir(tiles_xdpath)
    for i, tilename in enumerate(tilenames):
        #if i > 0: break
        demvfill_unspervised(outdir,tiles_xdpath,tilename,method)

    tf = time.perf_counter() -ti 
    print(f'run.time ={tf/60} min(s)')
    print(f'method::{method} numtiles {len(tilenames)}')
   


            
            

        
        
