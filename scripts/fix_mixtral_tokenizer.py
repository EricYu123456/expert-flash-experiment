#!/usr/bin/env python3
"""Restore Mixtral's declared Metaspace prefix rule in converted GGUF metadata.

The pinned converter treats tokenizer_config.json's null add_prefix_space as
false. This checkpoint's tokenizer.json uses Metaspace prepend_scheme=first,
which agrees with its SentencePiece model and Hugging Face tokenizer. Only the
existing one-byte GGUF boolean is repaired; tensor data and sources are untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-model', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--llama-source', type=Path, default=Path(
        '/home/nfs_client/ericyu/expert_flash_experiment/sources/llama.cpp-927dada6c'))
    a = p.parse_args()
    config = json.loads((a.source_model/'config.json').read_text())
    tokenizer_config = json.loads((a.source_model/'tokenizer_config.json').read_text())
    tokenizer = json.loads((a.source_model/'tokenizer.json').read_text())
    assert config['model_type']=='mixtral'
    assert tokenizer_config['add_prefix_space'] is None
    assert tokenizer['pre_tokenizer']['type']=='Metaspace'
    assert tokenizer['pre_tokenizer']['prepend_scheme']=='first'
    sys.path.insert(0,str(a.llama_source/'gguf-py'))
    from gguf import GGUFReader, GGUFValueType
    reader = GGUFReader(str(a.model))
    field = reader.fields['tokenizer.ggml.add_space_prefix']
    assert field.types == [GGUFValueType.BOOL]
    old = field.contents()
    if a.check_only:
        assert old is True, 'GGUF prefix metadata differs from the source tokenizer'
        print('Mixtral GGUF prefix metadata matches source tokenizer')
        return
    if old is True:
        print('Mixtral prefix metadata already correct')
        return
    before = reader.data[:reader.data_offset].tobytes()
    offset = int(field.parts[-1].ctypes.data-reader.data.ctypes.data)
    assert 0 <= offset < reader.data_offset and before[offset]==0
    size = a.model.stat().st_size
    layout = [(t.name,t.data_offset,t.n_bytes,t.shape.tolist()) for t in reader.tensors]
    with a.model.open('r+b') as file:
        file.seek(offset)
        assert file.read(1)==b'\x00'
        file.seek(offset)
        file.write(b'\x01')
        file.flush()
        os.fsync(file.fileno())
    updated = GGUFReader(str(a.model))
    after = updated.data[:updated.data_offset].tobytes()
    assert [i for i,(x,y) in enumerate(zip(before,after)) if x!=y]==[offset]
    assert updated.fields[field.name].contents() is True
    assert layout==[(t.name,t.data_offset,t.n_bytes,t.shape.tolist()) for t in updated.tensors]
    assert a.model.stat().st_size==size
    result = dict(source_directory=str(a.source_model.resolve()), source_files_unchanged=True,
                  model=str(a.model.resolve()), field=field.name, old_value=False, new_value=True,
                  byte_offset=offset, tensor_data_offset=updated.data_offset,
                  changed_header_bytes=1, tensor_layout_unchanged=True, tensor_payload_not_written=True,
                  header_sha256_before=hashlib.sha256(before).hexdigest(),
                  header_sha256_after=hashlib.sha256(after).hexdigest(),
                  source_rule='tokenizer.json Metaspace prepend_scheme=first; null config does not disable it',
                  full_model_sha256_status='Recompute with inspect_gguf.py after metadata repair')
    if a.output:
        a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
