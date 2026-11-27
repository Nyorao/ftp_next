"""Build label-free layouts from a verified image-only provenance audit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import SimpleITK as sitk

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from freqssl.geometry import sagittal_plan, transformed_geometry


def header(path):
    reader=sitk.ImageFileReader();reader.SetFileName(str(path));reader.ReadImageInformation()
    return dict(size_xyz=list(reader.GetSize()),spacing_xyz_mm=list(reader.GetSpacing()),
                origin_xyz=list(reader.GetOrigin()),direction=list(reader.GetDirection()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    mapping_path=args.audit_dir/'verified_image_mapping.json'
    mapping=json.loads(mapping_path.read_text())
    fingerprints=json.loads((args.audit_dir/'image_fingerprints.json').read_text())
    original={r['path']:r for r in fingerprints['original']}
    target_hashes={r['case']:r['array_sha256'] for r in fingerprints['converted']}
    records=[]
    for row in mapping:
        if len(row['matches'])!=1 or len(row['patient_candidates'])!=1:
            raise ValueError(f"Ambiguous source: {row['case']}")
        match=row['matches'][0]
        if not match['exact_image_voxel_match'] or len(match['transforms'])!=1:
            raise ValueError('Need one verified image transform')
        t=match['transforms'][0]; raw=header(match['path']); provided=header(row['target'])
        correct=transformed_geometry(raw['size_xyz'],raw['spacing_xyz_mm'],raw['origin_xyz'],raw['direction'],
                                     t['transpose_zyx'],t['flip_axes_zyx'])
        if correct['size_xyz']!=provided['size_xyz']:raise ValueError('Transformed shape mismatch')
        changes=[key for key in correct if not np.allclose(correct[key],provided[key],atol=1e-6,rtol=0)]
        records.append({'case':row['case'],'split':row['split'],'patient_id':match['patient'],
                        'source_image':match['path'],'converted_image':row['target'],
                        'source_image_array_sha256':original[match['path']]['array_sha256'],
                        'converted_image_array_sha256':target_hashes[row['case']],
                        'verified_transform':t,'provided_geometry':provided,'verified_geometry':correct,
                        'metadata_fields_requiring_correction':changes,'sagittal_plan':sagittal_plan(correct)})
    patients={split:{r['patient_id'] for r in records if r['split']==split} for split in ('train','valid','test')}
    overlaps={f'{a}/{b}':sorted(patients[a]&patients[b]) for a,b in [('train','valid'),('train','test'),('valid','test')]}
    if any(overlaps.values()):raise ValueError(f'Patient leakage: {overlaps}')
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'source_mapping':str(mapping_path.resolve()),
             'source_mapping_sha256':hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
             'labels_read':False,'pixel_data_read_by_this_manifest_builder':False,
             'identity_evidence':'Prior audit confirms complete image voxel equality after recorded axis transformation',
             'source_data_modified':False,'patient_overlap':overlaps,
             'volumes':len(records),'metadata_mismatch_cases':[r['case'] for r in records if r['metadata_fields_requiring_correction']],
             'use':'New label-free geometry input pipeline; not applied to running frozen legacy-PNG experiments',
             'records':records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='records'},indent=2))


if __name__=='__main__':main()
