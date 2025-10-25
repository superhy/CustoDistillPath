import argparse

from support.files import (
    move_TCGA_download_file_rename_batch_from_barcode_table,
    parse_filesystem_slide,
)
from wsi.process import slide_tiles_split_keep_object


DEFAULT_SLIDE_DIR = '/exafs1/well/rittscher/projects/TCGA-COAD/data/slides'
DEFAULT_TILE_PKL_DIR = '/exafs1/well/rittscher/projects/TCGA-COAD/data/tilelists'
DEFAULT_ORIGINAL_DATA_DIR = '/exafs1/well/rittscher/shared/datasets/TCGA-GI/CR'
DEFAULT_BARCODE_TABLE = 'metadata/TCGA_COAD/nationwidechildrens.org_clinical_patient_coad.txt'
DEFAULT_MODE = 'copy'
DEFAULT_BARCODE_COLUMN = 'bcr_patient_barcode'


def _run_split_tiles(args: argparse.Namespace) -> None:
    slide_tiles_split_keep_object(
        slide_dir=args.slide_dir,
        tile_pkl_dir=args.tile_pkl_dir,
    )


def _run_move_slides(args: argparse.Namespace) -> None:
    slide_paths = parse_filesystem_slide(args.original_data_dir)
    print(
        'Identified {} slide files under {}.'.format(
            len(slide_paths), args.original_data_dir
        )
    )

    move_TCGA_download_file_rename_batch_from_barcode_table(
        tcga_slide_path_list=slide_paths,
        parse_data_slide_dir=args.output_dir,
        barcode_table_path=args.barcode_table,
        mode=args.mode,
        barcode_column=args.barcode_column,
        table_sep=args.table_sep,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='CLI entry point for running slide processing utilities.'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    split_parser = subparsers.add_parser(
        'split-tiles',
        help='Generate tile object pickles for every slide in a directory.',
    )
    split_parser.set_defaults(func=_run_split_tiles)
    split_parser.add_argument(
        '--slide-dir',
        default=DEFAULT_SLIDE_DIR,
        help='Directory containing the source slide files.',
    )
    split_parser.add_argument(
        '--tile-pkl-dir',
        default=DEFAULT_TILE_PKL_DIR,
        help='Directory where tile object pickle files will be written.',
    )

    mover_parser = subparsers.add_parser(
        'move-slides',
        help='Move or copy TCGA slides filtered by a barcode table.',
    )
    mover_parser.set_defaults(func=_run_move_slides)
    mover_parser.add_argument(
        '--original-data-dir',
        default=DEFAULT_ORIGINAL_DATA_DIR,
        help='Directory containing the downloaded TCGA slide files.',
    )
    mover_parser.add_argument(
        '--output-dir',
        default=DEFAULT_SLIDE_DIR,
        help='Destination directory for the renamed slides.',
    )
    mover_parser.add_argument(
        '--barcode-table',
        default=DEFAULT_BARCODE_TABLE,
        help='Path to the table that includes case ids.',
    )
    mover_parser.add_argument(
        '--mode',
        choices=['move', 'copy'],
        default=DEFAULT_MODE,
        help='Whether to move or copy the slide files.',
    )
    mover_parser.add_argument(
        '--barcode-column',
        default=DEFAULT_BARCODE_COLUMN,
        help='Column name inside the table that stores the case ids.',
    )
    mover_parser.add_argument(
        '--table-sep',
        default=None,
        help='Optional delimiter for delimited text tables (e.g., ",", "\\t").',
    )

    return parser


def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
