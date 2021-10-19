#!/usr/bin/env python3
############################################################
# Author: Forrest Williams                                 #
# Worflow Creator: Scott Staniewicz                        #
############################################################

import os
import sys
import argparse
import subprocess
from pathlib import Path

import numpy as np

################################################################################################

conf_template = '''
    # snaphu configuration file

    # Input file name
    INFILE       {infile}
    INFILEFORMAT        {informat}

    # Input file line length
    LINELENGTH    {width}


    # Output file
    OUTFILE             {outfile}
    OUTFILEFORMAT       {outformat}

    # Correlation file
    CORRFILE     {corfile}
    CORRFILEFORMAT      FLOAT_DATA

    STATCOSTMODE SMOOTH

    INITMETHOD    MCF

    ################
    # Tile control #
    ################

    # Parameters in this section describe how the input files will be 
    # tiled.  This is mainly used for tiling, in which different
    # patches of the interferogram are unwrapped separately.

    # Number of rows and columns of tiles into which the data files are
    # to be broken up.
    NTILEROW		{ntilerow}
    NTILECOL		{ntilecol}

    # Overlap, in pixels, between neighboring tiles.
    # Using the same overlap for rows and cols here, but they can be different in general
    ROWOVRLP		{overlap}
    COLOVRLP		{overlap}

    # Maximum number of child processes to start for parallel tile
    # unwrapping.
    NPROC			{nproc}
    '''


mask_template = '''
    ###########
    # Masking #
    ###########

    # Input file of signed binary byte (signed char) values.  Values in
    # the file should be either 0 or 1, with 0 denoting interferogram
    # pixels that should be masked out and 1 denoting valid pixels.  The
    # array should have the same dimensions as the input wrapped phase
    # array.
    BYTEMASKFILE	{mask}
    '''


unwrap_template = """
    UNWRAPPED_IN    TRUE
    UNWRAPPEDINFILEFORMAT    FLOAT_DATA
    """

################################################################################################
REFERENCE = '''source:
  Script author: Forrest Williams
  Origional workflow: Scott Staniewicz
'''

EXAMPLE = '''example:
  UAVSAR_SNAPHU_2stage.py -i filt_20090220_20091119.int -c filt_20090220_20091119.cor -s 4892 8333 -m waterMask.rdr
  UAVSAR_SNAPHU_2stage.py -i filt_20090220_20091119.int -c filt_20090220_20091119.cor -s 4892 8333 --looks 3 3 --overlap 400
'''


def create_parser():
    parser = argparse.ArgumentParser(description='Two stage unwrapping of UAVSAR ifg with SNAPHU.',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=REFERENCE+'\n'+EXAMPLE)

    parser.add_argument('-i', dest='ifg_file', help='Interferogram file to be unwrapped')
    parser.add_argument('-c', dest='cor_file', help='Correlation file')
    parser.add_argument('-s', dest='shape', nargs=2, type=int, help='Shape of interferogram e.g. (ROWS COLS)')
    
    parser.add_argument('-m', dest='mask_file', default=None, help='Water mask file.')
    
    parser.add_argument('--tiles', dest='tiles', nargs=2, default=[3,3], type=int,
                        help='Number of tiles in for SNAPHU processing (ROWS COLS)')
    parser.add_argument('--looks', dest='looks', nargs=2, default=[3,3], type=int,
                        help='Number of looks for initial unwrapping (ROWS COLS)')
    parser.add_argument('--overlap', dest='overlap', default=400, type=int, help='Overlap for tile processing')
    
    return parser


def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    
    inps.ifg_file = Path(inps.ifg_file).resolve()
    inps.cor_file = Path(inps.cor_file).resolve()
    
    if inps.mask_file:
        inps.mask_file = Path(inps.mask_file).resolve()
    
    inps.home = inps.ifg_file.parent
    
    return inps

################################################################################################
def take_looks(arr, row_looks, col_looks, separate_complex=False, **kwargs):
    '''Downsample a numpy matrix by summing blocks of (row_looks, col_looks)

    Cuts off values if the size isn't divisible by num looks

    NOTE: For complex data, looks on the magnitude are done separately
    from looks on the phase

    Args:
        arr (ndarray) 2D array of an image
        row_looks (int) the reduction rate in row direction
        col_looks (int) the reduction rate in col direction
        separate_complex (bool): take looks on magnitude and phase separately
            Better to preserve the look of the magnitude

    Returns:
        ndarray, size = ceil(rows / row_looks, cols / col_looks)
    '''
    if row_looks == 1 and col_looks == 1:
        return arr
    if np.iscomplexobj(arr) and separate_complex:
        mag_looked = take_looks(np.abs(arr), row_looks, col_looks)
        phase_looked = take_looks(np.angle(arr), row_looks, col_looks)
        return mag_looked * np.exp(1j * phase_looked)

    rows, cols = arr.shape
    new_rows = rows // row_looks
    new_cols = cols // col_looks

    row_cutoff = rows % row_looks
    col_cutoff = cols % col_looks

    if row_cutoff != 0:
        arr = arr[:-row_cutoff, :]
    if col_cutoff != 0:
        arr = arr[:, :-col_cutoff]
    # For taking the mean, treat integers as floats
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype('float')

    return np.mean(arr.reshape(new_rows, row_looks, new_cols, col_looks), axis=(3, 1))


def apply_phasemask(unw_low, intf_high):
    '''Apply the integer phase ambiguity in the unwrapped phase array `unw_low` 
    to the phase in the wrapped, complex array `intf_high`. 
    
    You should use this routine to apply an unwrapping solution obtained on 
    low resolution data back to a higher resolution version of the same data.

    If the `unw_low` has been multilooked 3x3 extra times beyond `intf_high`, then
    each pixel in `unw_low` will be used for a 3x3 square of high-res data.
    
    Returns:
        ndarray, size = `intf_high.shape`. Unwrapped version of the high resolution data.
    '''
    from skimage.transform import resize
    # Resize the low res data to the same size as high-res. 
    # mode='constant' uses nearest-neighbor interpolation
    unw_high = resize(unw_low, intf_high.shape, mode='constant', anti_aliasing=False)

    # Make sure we're working with phase, not complex numbers
    highres = np.angle(intf_high) if np.iscomplexobj(intf_high) else intf_high

    # Find the total radians between the wrapped an unwrapped 
    dx = highres - unw_high
    # convert the to nearest number of whol cycles, then back to ambiguity in radians
    whole_cycles = np.around(dx / (2 * np.pi))
    ambig = (2 * np.pi) * whole_cycles
    highres_unw = highres - ambig
    
    return highres_unw
    # To return the unwrapped with both amplitude an phase (to save as a band-interleaved file)
    # return np.stack((np.abs(intf_high), highres), axis=0)


def write_snaphu_config(inps,in_ifg,in_cor,shape,out_conf,out_file,in_mask=None,unwrapped=False):
    '''Writes a configuration file that is readable by SNAPHU'''
    ntilerow, ntilecol = inps.tiles
    nproc = ntilerow * ntilecol
    
    in_format = 'FLOAT_DATA' if unwrapped else 'COMPLEX_DATA'

    options = dict(
        infile=in_ifg,
        corfile=in_cor,
        width=shape[1],
        outfile=out_file,
        informat=in_format,
        outformat='FLOAT_DATA',
        overlap=inps.overlap,
        ntilerow=ntilerow,
        ntilecol=ntilecol,
        nproc=nproc)
    
    if in_mask:
        template = conf_template + mask_template
        options['mask'] = in_mask
    else:
        template = conf_template

    if unwrapped:
        template += unwrap_template

    template = template.format(**options)

    with open(out_conf, "w") as f:
        f.write(template)

    return out_conf, out_file


################################################################################################
def main(iargs=None):
    inps = cmd_line_parse(iargs)
    
    
    # Multilook files
    ifg = np.fromfile(inps.ifg_file, dtype='complex64').reshape((-1, inps.shape[0]))
    cor = np.fromfile(inps.cor_file, dtype='float32').reshape((-1, inps.shape[0]))

    ifg_extra_looked = take_looks(ifg, *inps.looks)
    cor_extra_looked = take_looks(cor, *inps.looks)
    
    look_prefix = 'looked_{}x{}_'.format(*inps.looks)
    
    ifg_extra_looked_file = inps.home / (look_prefix + inps.ifg_file.name)
    cor_extra_looked_file = inps.home / (look_prefix + inps.cor_file.name)
    
    unlooked_shape = ifg.shape
    looked_shape = ifg_extra_looked.shape

    print('Saving looked ifg to', ifg_extra_looked_file)
    ifg_extra_looked.tofile(ifg_extra_looked_file)
    
    print('Saving looked cor to', cor_extra_looked_file)
    cor_extra_looked.tofile(cor_extra_looked_file)

    if inps.mask_file:
        mask = np.fromfile(inps.mask_file, dtype='byte').reshape((-1, inps.shape[0]))
        mask_extra_looked = (take_looks(mask, *inps.looks)).astype('byte')
        
        mask_extra_looked_file = inps.home / (look_prefix + inps.mask_file.name)

        print('Saving looked mask to', mask_extra_looked_file)
        mask_extra_looked.tofile(mask_extra_looked_file)
    
    # Perform looked unwrapping
    opts = dict(inps = inps,
              in_ifg = ifg_extra_looked_file,
              in_cor = cor_extra_looked_file,
              shape = looked_shape,
              out_conf = inps.home/'snaphu_looked.conf',
              out_file = inps.home/ifg_extra_looked_file.name.replace('int','unw'))
    
    if inps.mask_file:
        opts['in_mask'] = mask_extra_looked_file

    conf_looked, unw_extra_looked_file = write_snaphu_config(**opts)

    print('Outputting looked unwrapped phase to', unw_extra_looked_file)
    cmd = f'snaphu -f {conf_looked} |& tee -i {inps.home / "loooked.log"}'
    print(cmd)
    subprocess.call(cmd, shell=True)

    # Transfer 2pi ambiguity
    unw_extra_looked = np.fromfile(unw_extra_looked_file, dtype='float32').reshape(looked_shape)
    unw_2stage = apply_phasemask(unw_extra_looked, ifg)

    # Save as a float32 file
    unw_phasemask_file = inps.home / ('2stage_' + unw_extra_looked_file.name)
    
    print('Outputting looked unwrapped phase to', unw_extra_looked_file)
    unw_2stage.tofile(unw_phasemask_file)
    
    # Reoptimize unwrapping
    opts = dict(inps = inps,
              in_ifg = unw_phasemask_file,
              in_cor = inps.cor_file,
              shape = unlooked_shape,
              out_conf = inps.home/'snaphu_reopt.conf',
              out_file = inps.home/('reopt_' + inps.ifg_file.name.replace('int','unw')),
              unwrapped = True)

    if inps.mask_file:
        opts['in_mask'] =inps.mask_file

    conf_reopt, unw_reopt_file = write_snaphu_config(**opts)
    
    print('Outputting reoptimized output to', unw_reopt_file)
    cmd = f'snaphu -f {conf_reopt} |& tee -i {inps.home / "reoptimized.log"}'
    print(cmd)
    subprocess.call(cmd, shell=True)
    
    print('Done.')
    return unw_reopt_file


################################################################################################
if __name__ == '__main__':
    main(sys.argv[1:])
    