#!/usr/bin/python

import os
import subprocess as sp
import shutil as sh
import argparse

parser = argparse.ArgumentParser(
    description = 'Copies and converts drawio XML and PDF files from Dropbox'
)
parser.add_argument('-s', '--src_dir',
    help = 'The src directory containing the drawio XML and PDF files',
    default = '/Users/sweet/Dropbox/Apps/drawio'
)
parser.add_argument('-v', '--verbose',
    help = 'Specify this to get verbose output',
    action = 'store_true',
    default = False
)
args = parser.parse_args()

if not os.path.isdir(args.src_dir):
    raise IOError('Source directory ' + args.src_dir + ' is not a directory')

# top level destination is the same as this file
dst_dir = os.path.dirname(os.path.realpath(__file__))

# edit these to copy/convert files
file_pre = 'fcdiff'
file_dict = {
    'methods' : (
        'graphical_model',
        'graphical_model_rt',
        'graphical_model_all',
    ),
}

for sub_dir, sub_files in file_dict.items():
    sub_dst_dir = os.path.join(dst_dir, sub_dir)

    for fs in sub_files:
        src_stem = os.path.join(args.src_dir, file_pre + '-' + sub_dir + '-' + fs)
        dst_stem = os.path.join(sub_dst_dir, fs)

        if args.verbose:
            print('Copying XMl source to destination')
        xml_src = src_stem + '.xml'
        xml_dst = dst_stem + '.xml'
        if not os.path.isfile(xml_src):
            raise IOError('Could not find drawio XML file ' + xml_src)
        sh.copy2(xml_src, xml_dst)

        if args.verbose:
            print('Copying PDF source to destination')
        pdf_src = src_stem + '.pdf'
        pdf_dst = dst_stem + '.pdf'
        if not os.path.isfile(pdf_src):
            raise IOError('Could not find drawio PDF file ' + pdf_src)

        sh.copy2(pdf_src, pdf_dst)

        if args.verbose:
            print('Converting PDF to SVG')
        svg_dst = dst_stem + '.svg'
        sp.check_call(['pdf2svg', pdf_dst, svg_dst])

