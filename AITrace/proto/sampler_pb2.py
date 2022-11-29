# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/sampler.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='proto/sampler.proto',
  package='AITrace',
  syntax='proto2',
  serialized_options=_b('\n\013com.AITraceB\014SamplerProtoP\001Z\tAITracepb'),
  serialized_pb=_b('\n\x13proto/sampler.proto\x12\x07\x41ITrace\"\x99\x02\n\x07Sampler\x12\x14\n\ninput_feed\x18\x01 \x01(\x08H\x00\x12\x13\n\ttrain_set\x18\x02 \x01(\x08H\x00\x12\x18\n\x0evalidation_set\x18\x03 \x01(\x08H\x00\x12\x19\n\x0fserver_sampling\x18\x04 \x01(\x08H\x00\x12\x17\n\x0fprediction_type\x18\x05 \x01(\t\x12\x12\n\nbatch_size\x18\x07 \x01(\x05\x12\x1a\n\x12temperature_micros\x18\t \x01(\x05\x12\x13\n\x0bserver_port\x18\n \x01(\x05\x12\x41\n\x14termination_criteria\x18\x0b \x03(\x0b\x32#.AITrace.SampleTerminationCriterionB\r\n\x0bsample_feed\"\x86\x01\n\x1aSampleTerminationCriterion\x12)\n\x06maxlen\x18\x01 \x01(\x0b\x32\x17.AITrace.MaxTokenLengthH\x00\x12\x30\n\x06symtok\x18\x02 \x01(\x0b\x32\x1e.AITrace.SymmetricalTokenDepthH\x00\x42\x0b\n\tcriterion\"2\n\x0eMaxTokenLength\x12 \n\x18maximum_tokens_in_sample\x18\x01 \x01(\x05\"S\n\x15SymmetricalTokenDepth\x12\x1c\n\x14\x64\x65pth_increase_token\x18\x01 \x01(\t\x12\x1c\n\x14\x64\x65pth_decrease_token\x18\x02 \x01(\tB(\n\x0b\x63om.AITraceB\x0cSamplerProtoP\x01Z\tAITracepb')
)




_SAMPLER = _descriptor.Descriptor(
  name='Sampler',
  full_name='AITrace.Sampler',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='input_feed', full_name='AITrace.Sampler.input_feed', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='train_set', full_name='AITrace.Sampler.train_set', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='validation_set', full_name='AITrace.Sampler.validation_set', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='server_sampling', full_name='AITrace.Sampler.server_sampling', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='prediction_type', full_name='AITrace.Sampler.prediction_type', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='batch_size', full_name='AITrace.Sampler.batch_size', index=5,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='temperature_micros', full_name='AITrace.Sampler.temperature_micros', index=6,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='server_port', full_name='AITrace.Sampler.server_port', index=7,
      number=10, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='termination_criteria', full_name='AITrace.Sampler.termination_criteria', index=8,
      number=11, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='sample_feed', full_name='AITrace.Sampler.sample_feed',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=33,
  serialized_end=314,
)


_SAMPLETERMINATIONCRITERION = _descriptor.Descriptor(
  name='SampleTerminationCriterion',
  full_name='AITrace.SampleTerminationCriterion',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='maxlen', full_name='AITrace.SampleTerminationCriterion.maxlen', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='symtok', full_name='AITrace.SampleTerminationCriterion.symtok', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='criterion', full_name='AITrace.SampleTerminationCriterion.criterion',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=317,
  serialized_end=451,
)


_MAXTOKENLENGTH = _descriptor.Descriptor(
  name='MaxTokenLength',
  full_name='AITrace.MaxTokenLength',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='maximum_tokens_in_sample', full_name='AITrace.MaxTokenLength.maximum_tokens_in_sample', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=453,
  serialized_end=503,
)


_SYMMETRICALTOKENDEPTH = _descriptor.Descriptor(
  name='SymmetricalTokenDepth',
  full_name='AITrace.SymmetricalTokenDepth',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='depth_increase_token', full_name='AITrace.SymmetricalTokenDepth.depth_increase_token', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='depth_decrease_token', full_name='AITrace.SymmetricalTokenDepth.depth_decrease_token', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=505,
  serialized_end=588,
)

_SAMPLER.fields_by_name['termination_criteria'].message_type = _SAMPLETERMINATIONCRITERION
_SAMPLER.oneofs_by_name['sample_feed'].fields.append(
  _SAMPLER.fields_by_name['input_feed'])
_SAMPLER.fields_by_name['input_feed'].containing_oneof = _SAMPLER.oneofs_by_name['sample_feed']
_SAMPLER.oneofs_by_name['sample_feed'].fields.append(
  _SAMPLER.fields_by_name['train_set'])
_SAMPLER.fields_by_name['train_set'].containing_oneof = _SAMPLER.oneofs_by_name['sample_feed']
_SAMPLER.oneofs_by_name['sample_feed'].fields.append(
  _SAMPLER.fields_by_name['validation_set'])
_SAMPLER.fields_by_name['validation_set'].containing_oneof = _SAMPLER.oneofs_by_name['sample_feed']
_SAMPLER.oneofs_by_name['sample_feed'].fields.append(
  _SAMPLER.fields_by_name['server_sampling'])
_SAMPLER.fields_by_name['server_sampling'].containing_oneof = _SAMPLER.oneofs_by_name['sample_feed']
_SAMPLETERMINATIONCRITERION.fields_by_name['maxlen'].message_type = _MAXTOKENLENGTH
_SAMPLETERMINATIONCRITERION.fields_by_name['symtok'].message_type = _SYMMETRICALTOKENDEPTH
_SAMPLETERMINATIONCRITERION.oneofs_by_name['criterion'].fields.append(
  _SAMPLETERMINATIONCRITERION.fields_by_name['maxlen'])
_SAMPLETERMINATIONCRITERION.fields_by_name['maxlen'].containing_oneof = _SAMPLETERMINATIONCRITERION.oneofs_by_name['criterion']
_SAMPLETERMINATIONCRITERION.oneofs_by_name['criterion'].fields.append(
  _SAMPLETERMINATIONCRITERION.fields_by_name['symtok'])
_SAMPLETERMINATIONCRITERION.fields_by_name['symtok'].containing_oneof = _SAMPLETERMINATIONCRITERION.oneofs_by_name['criterion']
DESCRIPTOR.message_types_by_name['Sampler'] = _SAMPLER
DESCRIPTOR.message_types_by_name['SampleTerminationCriterion'] = _SAMPLETERMINATIONCRITERION
DESCRIPTOR.message_types_by_name['MaxTokenLength'] = _MAXTOKENLENGTH
DESCRIPTOR.message_types_by_name['SymmetricalTokenDepth'] = _SYMMETRICALTOKENDEPTH
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Sampler = _reflection.GeneratedProtocolMessageType('Sampler', (_message.Message,), dict(
  DESCRIPTOR = _SAMPLER,
  __module__ = 'proto.sampler_pb2'
  # @@protoc_insertion_point(class_scope:AITrace.Sampler)
  ))
_sym_db.RegisterMessage(Sampler)

SampleTerminationCriterion = _reflection.GeneratedProtocolMessageType('SampleTerminationCriterion', (_message.Message,), dict(
  DESCRIPTOR = _SAMPLETERMINATIONCRITERION,
  __module__ = 'proto.sampler_pb2'
  # @@protoc_insertion_point(class_scope:AITrace.SampleTerminationCriterion)
  ))
_sym_db.RegisterMessage(SampleTerminationCriterion)

MaxTokenLength = _reflection.GeneratedProtocolMessageType('MaxTokenLength', (_message.Message,), dict(
  DESCRIPTOR = _MAXTOKENLENGTH,
  __module__ = 'proto.sampler_pb2'
  # @@protoc_insertion_point(class_scope:AITrace.MaxTokenLength)
  ))
_sym_db.RegisterMessage(MaxTokenLength)

SymmetricalTokenDepth = _reflection.GeneratedProtocolMessageType('SymmetricalTokenDepth', (_message.Message,), dict(
  DESCRIPTOR = _SYMMETRICALTOKENDEPTH,
  __module__ = 'proto.sampler_pb2'
  # @@protoc_insertion_point(class_scope:AITrace.SymmetricalTokenDepth)
  ))
_sym_db.RegisterMessage(SymmetricalTokenDepth)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
