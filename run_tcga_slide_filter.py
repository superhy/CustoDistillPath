import argparse

from support.files import (move_TCGA_download_file_rename_batch_from_barcode_table,
                           parse_filesystem_slide)


def main():
    parser = argparse.ArgumentParser(
        description='Move or copy TCGA slides filtered by a barcode table.')
    parser.add_argument('--original-data-dir', required=True,
                        help='Directory containing the downloaded TCGA slide files.')
    parser.add_argument('--output-dir', required=True,
                        help='Destination directory for the renamed slides.')
    parser.add_argument('--barcode-table', required=True,
                        help='Path to the table that includes case ids.')
    parser.add_argument('--mode', choices=['move', 'copy'], default='move',
                        help='Whether to move or copy the slide files.')
    parser.add_argument('--barcode-column', default='bcr_patient_barcode',
                        help='Column name inside the table that stores the case ids.')
    parser.add_argument('--table-sep', default=None,
                        help='Optional delimiter for delimited text tables (e.g., ",", "\\t").')
    args = parser.parse_args()

    # Set default values
    default_values = {
        'original_data_dir': '/exafs1/well/rittscher/shared/datasets/TCGA-GI/CR',
        'output_dir': '/exafs1/well/rittscher/projects/TCGA-COAD/data/slides',
        'barcode_table': 'metadata/TCGA_COAD/nationwidechildrens.org_clinical_patient_coad.txt',
        'mode': 'copy',
        'barcode_column': 'bcr_patient_barcode',
        'table_sep': None,
    }

    # Override defaults with command line arguments if provided
    for key, value in default_values.items():
        if getattr(args, key.replace('-', '_'), None) is None:
            setattr(args, key.replace('-', '_'), value)
    slide_paths = parse_filesystem_slide(args.original_data_dir)
    print('Identified {} slide files under {}.'.format(
        len(slide_paths), args.original_data_dir))

    move_TCGA_download_file_rename_batch_from_barcode_table(
        tcga_slide_path_list=slide_paths,
        parse_data_slide_dir=args.output_dir,
        barcode_table_path=args.barcode_table,
        mode=args.mode,
        barcode_column=args.barcode_column,
        table_sep=args.table_sep)


if __name__ == '__main__':
    main()
