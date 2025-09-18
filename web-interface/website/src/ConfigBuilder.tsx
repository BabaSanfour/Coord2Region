import { KeyboardEvent, useCallback, useEffect, useMemo, useState } from 'react';
import Form, { IChangeEvent } from '@rjsf/core';
import validator from '@rjsf/validator-ajv8';
import {
  FieldTemplateProps,
  FieldProps,
  RJSFSchema,
  UiSchema,
  WidgetProps
} from '@rjsf/utils';
import YAML from 'js-yaml';
import clsx from 'clsx';
import schemaSource from '../../../docs/static/schema.json';
import './configBuilder.css';

type InputMode = 'coords' | 'region_names';

type CoordEntryMode = 'paste' | 'file';

type FormState = Record<string, unknown>;

type ParsedCoordinates = {
  coords: number[][];
  errors: string[];
};

type SchemaProperty = {
  default?: unknown;
  title?: string;
  description?: string;
  type?: string | string[];
  anyOf?: Array<Record<string, unknown>>;
};

const schema = schemaSource as RJSFSchema;

const isHttpUrl = (value: string) => /^https?:\/\//i.test(value.trim());

const looksLikePath = (value: string) => {
  const trimmed = value.trim();
  if (!trimmed) {
    return false;
  }
  if (trimmed.startsWith('~') || trimmed.startsWith('./') || trimmed.startsWith('../')) {
    return true;
  }
  if (/^[A-Za-z]:\\/.test(trimmed)) {
    return true;
  }
  if (trimmed.includes('/') || trimmed.includes('\\')) {
    return true;
  }
  return false;
};

const atlasConfigFromValue = (value: string): Record<string, string> | null => {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  if (isHttpUrl(trimmed)) {
    return { atlas_url: trimmed };
  }
  if (looksLikePath(trimmed)) {
    return { atlas_file: trimmed };
  }
  return null;
};

const deriveAtlasConfigs = (names: unknown): Record<string, Record<string, string>> => {
  if (!Array.isArray(names)) {
    return {};
  }
  return names.reduce((acc, entry) => {
    if (typeof entry !== 'string') {
      return acc;
    }
    const config = atlasConfigFromValue(entry);
    if (config) {
      acc[entry] = config;
    }
    return acc;
  }, {} as Record<string, Record<string, string>>);
};

type AtlasOptionGroup = {
  id: string;
  label: string;
  options: string[];
};

const atlasGroups: AtlasOptionGroup[] = [
  {
    id: 'volumetric-nilearn',
    label: 'Volumetric (nilearn)',
    options: ['aal', 'basc', 'brodmann', 'destrieux', 'harvard-oxford', 'juelich', 'pauli', 'schaefer', 'talairach', 'yeo']
  },
  {
    id: 'surface-mne',
    label: 'Surface (mne)',
    options: [
      'aparc',
      'aparc.a2005s',
      'aparc.a2009s',
      'aparc_sub',
      'human-connectum project',
      'oasis.chubs',
      'pals_b12_lobes',
      'pals_b12_orbitofrontal',
      'pals_b12_visuotopic',
      'yeo2011'
    ]
  },
  {
    id: 'coordinates-mne',
    label: 'Coordinates (mne)',
    options: ['dosenbach', 'power', 'seitzman']
  }
];

const knownAtlases = new Set<string>(atlasGroups.flatMap((group) => group.options));

const deepClone = <T,>(value: T): T => JSON.parse(JSON.stringify(value));

const datasetSourceOptions = ["neurosynth", "neuroquery", "nidm_pain"] as const;
const outputFormatOptions = ["json", "pickle", "csv", "pdf", "directory"] as const;
type SelectOption = { value: string; label: string };

const summaryModelOptions: ReadonlyArray<SelectOption> = [
  { value: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash (Google)' },
  { value: 'gemini-1.5-pro', label: 'Gemini 1.5 Pro (Google)' },
  { value: 'gemini-1.0-pro', label: 'Gemini 1.0 Pro (Google)' },
  { value: 'claude-3-haiku', label: 'Claude 3 Haiku (Anthropic)' },
  { value: 'claude-3-opus', label: 'Claude 3 Opus (Anthropic)' },
  { value: 'deepseek-r1', label: 'DeepSeek R1 (OpenRouter)' },
  { value: 'deepseek-chat-v3-0324', label: 'DeepSeek Chat v3 (OpenRouter)' },
  { value: 'gpt-4', label: 'GPT-4 (OpenAI)' },
  { value: 'distilgpt2', label: 'distilgpt2 (Hugging Face)' }
];

const promptTypeOptions: ReadonlyArray<SelectOption> = [
  { value: 'summary', label: 'Integrated summary' },
  { value: 'region_name', label: 'Region name focus' },
  { value: 'function', label: 'Functional profile' },
  { value: 'custom', label: 'Custom prompt' }
];


const atlasProperty = (() => {
  const property = schema.properties?.atlas_names;
  if (!property || typeof property !== 'object') {
    return undefined;
  }
  const cloned = deepClone(property);
  if (Array.isArray((cloned as RJSFSchema).anyOf)) {
    const arrayOption = (cloned as RJSFSchema).anyOf?.find((option) => option?.type === 'array');
    if (arrayOption && typeof arrayOption === 'object') {
      Object.assign(cloned, arrayOption);
    }
    delete (cloned as RJSFSchema).anyOf;
  }
  if (!(cloned as RJSFSchema).items) {
    (cloned as RJSFSchema).items = { type: 'string' };
  }
  (cloned as RJSFSchema).type = 'array';
  return cloned as RJSFSchema;
})();

const sourcesProperty = (() => {
  // Align with schema which defines `sources` (not `dataset_sources`)
  const property = schema.properties?.sources;
  if (!property || typeof property !== 'object') {
    return undefined;
  }
  const cloned = deepClone(property) as RJSFSchema;
  if (Array.isArray(cloned.anyOf)) {
    const arrayOption = cloned.anyOf.find((option) => option?.type === 'array');
    if (arrayOption && typeof arrayOption === 'object') {
      Object.assign(cloned, arrayOption);
    }
    delete cloned.anyOf;
  }
  cloned.type = 'array';
  cloned.items = {
    type: 'string',
    enum: [...datasetSourceOptions]
  };
  cloned.uniqueItems = true;
  if (!Array.isArray(cloned.default)) {
    cloned.default = [];
  }
  return cloned as RJSFSchema;
})();

const outputFormatProperty = (() => {
  const property = schema.properties?.output_format;
  if (!property || typeof property !== 'object') {
    return undefined;
  }
  const cloned = deepClone(property) as RJSFSchema & { enum?: Array<string | null> };
  // Flatten anyOf to avoid RJSF "Option 1/2" selector
  delete (cloned as RJSFSchema).anyOf;
  // Allow null (represented by clearing selection) and restrict to known formats for validation
  cloned.type = ['string', 'null'] as unknown as RJSFSchema['type'];
  cloned.enum = [...outputFormatOptions, null];
  // Keep default null for empty state
  if (cloned.default === undefined) {
    cloned.default = null as unknown as RJSFSchema;
  }
  return cloned as RJSFSchema;
})();

const promptTypeProperty = (() => {
  const property = schema.properties?.prompt_type;
  if (!property || typeof property !== 'object') {
    return undefined;
  }
  const cloned = deepClone(property) as RJSFSchema & { enum?: Array<string | null> };
  delete cloned.anyOf;
  cloned.type = 'string';
  cloned.enum = promptTypeOptions.map((option) => option.value);
  if (!cloned.default) {
    cloned.default = 'summary';
  }
  return cloned as RJSFSchema;
})();

const dataDirProperty = (() => {
  const property = schema.properties?.data_dir;
  if (!property || typeof property !== 'object') {
    return undefined;
  }
  const cloned = deepClone(property) as RJSFSchema;
  delete cloned.anyOf;
  cloned.type = 'string';
  if (cloned.default === null) {
    cloned.default = '';
  }
  return cloned;
})();

const builderKeys: string[] = [
  'input_type',
  'data_dir',
  'sources',
  'atlas_names',
  'outputs',
  'output_format',
  'output_path',
  'studies',
  'study_sources',
  'study_radius',
  'study_limit',
  'prompt_type',
  'custom_prompt',
  'summary_model',
  'summary_max_tokens',
  'use_cached_dataset',
  'batch_size',
  'anthropic_api_key',
  'openai_api_key',
  'openrouter_api_key',
  'gemini_api_key',
  'huggingface_api_key',
  'email_for_abstracts'
];

const defaultRegionNames = (() => {
  const property = schema.properties?.region_names as SchemaProperty | undefined;
  if (!property) {
    return [] as string[];
  }
  const value = Object.prototype.hasOwnProperty.call(property, 'default')
    ? property.default
    : undefined;
  if (Array.isArray(value)) {
    return value
      .filter((name): name is string => typeof name === 'string' && name.trim().length > 0)
      .map((name) => name.trim());
  }
  return [] as string[];
})();

const deriveDefaults = (keys: string[]): FormState => {
  const defaults: FormState = {};
  const properties = schema.properties ?? {};
  keys.forEach((key) => {
    const prop = properties[key] as SchemaProperty | undefined;
    if (prop && Object.prototype.hasOwnProperty.call(prop, 'default')) {
      defaults[key] = prop.default as unknown;
    }
  });
  return defaults;
};

const builderSchema: RJSFSchema = {
  ...schema,
  properties: builderKeys.reduce((acc, key) => {
    const properties = schema.properties ?? {};
    if (properties[key]) {
      acc[key] = properties[key];
    }
    return acc;
  }, {} as NonNullable<RJSFSchema['properties']>),
  required: Array.isArray(schema.required)
    ? schema.required.filter((key) => builderKeys.includes(key))
    : undefined,
  additionalProperties: false
};

if (builderSchema.properties?.atlas_names && atlasProperty) {
  builderSchema.properties.atlas_names = atlasProperty;
}

if (builderSchema.properties?.sources && sourcesProperty) {
  builderSchema.properties.sources = sourcesProperty;
}

if (builderSchema.properties?.output_format && outputFormatProperty) {
  builderSchema.properties.output_format = outputFormatProperty;
}

if (builderSchema.properties?.prompt_type && promptTypeProperty) {
  builderSchema.properties.prompt_type = promptTypeProperty;
}

if (builderSchema.properties?.data_dir && dataDirProperty) {
  builderSchema.properties.data_dir = dataDirProperty;
}

const tooltipFromSchema = (key: string): string | undefined => {
  const property = (schema.properties ?? {})[key] as SchemaProperty | undefined;
  if (!property) {
    return undefined;
  }

  if (property.description) {
    return property.description;
  }

  const typeValues: string[] = [];
  const appendType = (value: SchemaProperty['type']) => {
    if (!value) {
      return;
    }
    if (Array.isArray(value)) {
      typeValues.push(...value.map(String));
    } else {
      typeValues.push(String(value));
    }
  };

  appendType(property.type);
  if (property.anyOf) {
    property.anyOf.forEach((option) => {
      appendType(option.type as SchemaProperty['type']);
    });
  }

  const uniqueTypes = Array.from(new Set(typeValues)).join(' | ');
  const title = property.title ? String(property.title) : key;
  const defaultValue = Object.prototype.hasOwnProperty.call(property, 'default')
    ? property.default
    : undefined;

  let tooltip = `${title}`;
  if (uniqueTypes) {
    tooltip += `\nType: ${uniqueTypes}`;
  }
  if (defaultValue !== undefined && defaultValue !== null) {
    tooltip += `\nDefault: ${JSON.stringify(defaultValue)}`;
  }
  return tooltip;
};

const tooltipMap: Record<string, string | undefined> = builderKeys.reduce(
  (acc, key) => {
    acc[key] = tooltipFromSchema(key);
    return acc;
  },
  {} as Record<string, string | undefined>
);

const FieldTemplate = (props: FieldTemplateProps) => {
  const { id, classNames, label, required, description, errors, help, children, hidden } = props;
  if (hidden) {
    return (
      <div className="form-field" style={{ display: 'none' }}>
        {children}
      </div>
    );
  }
  const key = id.replace(/^root_/, '').split('_')[0];
  const tooltip = tooltipMap[key];
  const labelText = label || key;

  return (
    <div className={clsx('form-field', classNames)}>
      {labelText && (
        <label
          htmlFor={id}
          className={clsx('field-label', tooltip && 'tooltip')}
          data-tooltip={tooltip}
        >
          {labelText}
          {required && <span className="required">*</span>}
        </label>
      )}
      {description}
      {children}
      {errors}
      {help}
    </div>
  );
};

const AtlasMultiSelect = (props: WidgetProps) => {
  const { id, value, disabled, readonly, onChange } = props;
  const rawValue = Array.isArray(value) ? (value as string[]) : [];
  const selected = Array.from(new Set(rawValue)).sort((a, b) => a.localeCompare(b));
  const isReadOnly = disabled || readonly;

  const commitSelection = (nextValues: Iterable<string>) => {
    const unique = Array.from(new Set(Array.from(nextValues))).sort((a, b) => a.localeCompare(b));
    onChange(unique.length ? unique : []);
  };

  const toggleAtlas = (atlas: string) => {
    const next = new Set(selected);
    if (next.has(atlas)) {
      next.delete(atlas);
    } else {
      next.add(atlas);
    }
    commitSelection(next);
  };

  const handleGroupToggle = (options: string[]) => {
    const next = new Set(selected);
    const hasMissing = options.some((option) => !next.has(option));
    if (hasMissing) {
      options.forEach((option) => next.add(option));
    } else {
      options.forEach((option) => next.delete(option));
    }
    commitSelection(next);
  };

  const [customAtlasInput, setCustomAtlasInput] = useState('');

  const addCustomAtlas = () => {
    if (isReadOnly) {
      return;
    }
    const atlas = customAtlasInput.trim();
    setCustomAtlasInput('');
    if (!atlas) {
      return;
    }
    const next = new Set(selected);
    next.add(atlas);
    commitSelection(next);
  };

  const handleCustomAtlasKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key !== 'Enter') {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    addCustomAtlas();
  };

  const customAtlases = selected.filter((atlas) => !knownAtlases.has(atlas));
  // Build display groups with options unique across groups (first occurrence wins)
  const baseGroups = (() => {
    const used = new Set<string>();
    return atlasGroups.map((group) => {
      const unique = [] as string[];
      for (const opt of Array.from(new Set(group.options))) {
        if (!used.has(opt)) {
          unique.push(opt);
          used.add(opt);
        }
      }
      return { ...group, options: unique };
    });
  })();
  const groups = customAtlases.length > 0
    ? [...baseGroups, { id: 'custom', label: 'Custom entries', options: customAtlases }]
    : baseGroups;

  return (
    <div className="atlas-widget" id={id}>
      <div className="atlas-grid">
        {groups.map((group) => {
          const groupSelected = group.options.filter((option) => selected.includes(option));
          const allSelected = group.options.length > 0 && groupSelected.length === group.options.length;
          const legendId = `${id}-${group.id}`;
          const selectAllLabel = allSelected
            ? `Clear all (${group.options.length})`
            : `Select all (${group.options.length})`;
          return (
            <div className="atlas-group" key={group.id}>
              <div className="atlas-group__header">
                <h5 id={legendId}>{group.label}</h5>
                {group.options.length > 0 && (
                  <div className="atlas-group__controls">
                    <span className="atlas-group__count">{groupSelected.length}/{group.options.length}</span>
                    <button
                      type="button"
                      className="atlas-group__action"
                      onClick={() => handleGroupToggle(group.options)}
                      disabled={isReadOnly}
                    >
                      {selectAllLabel}
                    </button>
                  </div>
                )}
              </div>
              <ul className="atlas-group__list" role="group" aria-labelledby={legendId}>
                {group.options.length === 0 ? (
                  <li className="atlas-group__item atlas-group__item--empty">Add a custom atlas to manage it here.</li>
                ) : (
                  group.options.map((option) => (
                    <li className="atlas-group__item" key={`${group.id}-${option}`}>
                      <label htmlFor={`${id}-${group.id}-${option}`}>
                        <input
                          id={`${id}-${group.id}-${option}`}
                          type="checkbox"
                          checked={selected.includes(option)}
                          onChange={() => toggleAtlas(option)}
                          disabled={isReadOnly}
                        />
                        <span>{option}</span>
                      </label>
                    </li>
                  ))
                )}
              </ul>
            </div>
          );
        })}
      </div>
      <div className="atlas-widget__form" role="group" aria-label="Custom atlas entry">
        <input
          type="text"
          name="customAtlas"
          value={customAtlasInput}
          onChange={(event) => setCustomAtlasInput(event.target.value)}
          onKeyDown={handleCustomAtlasKeyDown}
          placeholder="Add atlas (name, URL, or local path)"
          disabled={isReadOnly}
        />
        <button type="button" onClick={addCustomAtlas} disabled={isReadOnly}>
          Add
        </button>
      </div>
      <p className="atlas-summary">
        Selected {selected.length} atlas{selected.length === 1 ? '' : 'es'}.
      </p>
      {selected.length === 0 && <p className="atlas-widget__hint">Select one or more atlases to query.</p>}
    </div>
  );
};

const SummaryModelField = ({ formData, onChange, idSchema }: FieldProps) => {
  const value = typeof formData === 'string' ? formData : '';
  const inputId = idSchema?.$id ?? 'summary-model';
  const listId = `${inputId}-options`;

  return (
    <div className="form-field">
      <label className="field-label" htmlFor={inputId}>Summary Model</label>
      <input
        id={inputId}
        type="text"
        list={listId}
        value={value}
        placeholder="Start typing or pick a model"
        onChange={(event) => onChange(event.target.value || null)}
      />
      <datalist id={listId}>
        {summaryModelOptions.map((option) => (
          <option key={option.value} value={option.value}>{option.label}</option>
        ))}
      </datalist>
      <p className="helper">
        Choose a registered model or enter another identifier supported by your providers.
      </p>
    </div>
  );
};

const PromptTypeField = ({ formData, onChange, idSchema }: FieldProps) => {
  const value = typeof formData === 'string' && formData ? formData : 'summary';
  const inputId = idSchema?.$id ?? 'prompt-type';
  return (
    <div className="form-field">
      <label className="field-label" htmlFor={inputId}>Prompt Type</label>
      <select
        id={inputId}
        value={value}
        onChange={(event) => onChange(event.target.value || null)}
      >
        {promptTypeOptions.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      <p className="helper">
        Select a template for generated summaries. Choose “Custom prompt” to provide your own wording below.
      </p>
    </div>
  );
};

const OutputFormatField = ({ formData, onChange, idSchema }: FieldProps) => {
  const value = typeof formData === 'string' ? formData : '';
  const inputId = idSchema?.$id ?? 'output-format';
  return (
    <div className="form-field">
      <label className="field-label" htmlFor={inputId}>Output Format</label>
      <select
        id={inputId}
        value={value}
        onChange={(event) => {
          const next = event.target.value;
          onChange(next ? next : null);
        }}
      >
        <option value="">No export</option>
        {outputFormatOptions.map((option) => (
          <option key={option} value={option}>
            {option.toUpperCase()}
          </option>
        ))}
      </select>
      <p className="helper">Leave blank to skip file export.</p>
    </div>
  );
};

const CustomPromptField = ({ formData, onChange, formContext, idSchema }: FieldProps) => {
  const context = formContext as { promptType?: string } | undefined;
  if (context?.promptType !== 'custom') {
    return null;
  }
  const value = typeof formData === 'string' ? formData : '';
  const inputId = idSchema?.$id ?? 'custom-prompt';
  return (
    <div className="form-field">
      <label className="field-label" htmlFor={inputId}>Custom prompt template</label>
      <textarea
        id={inputId}
        rows={6}
        value={value}
        onChange={(event) => onChange(event.target.value || null)}
        placeholder="You are an expert neuroscientist..."
      />
      <p className="helper">
        Use {'{coord}'} for the coordinate and {'{studies}'} for the study list. These placeholders are filled automatically before calling the model.
      </p>
    </div>
  );
};

const widgets = {
  atlasMultiSelect: AtlasMultiSelect
};

const fields = {
  summaryModelField: SummaryModelField,
  promptTypeField: PromptTypeField,
  customPromptField: CustomPromptField,
  outputFormatField: OutputFormatField
};

const parseCoordinateText = (value: string): ParsedCoordinates => {
  const coords: number[][] = [];
  const errors: string[] = [];

  value
    .split(/\n+/)
    .map((line) => line.trim())
    .forEach((line, index) => {
      if (!line) {
        return;
      }
      const parts = line.split(/[,\s]+/).filter(Boolean);
      if (parts.length !== 3) {
        errors.push(`Line ${index + 1}: expected 3 numbers.`);
        return;
      }
      const tuple = parts.map((part) => Number(part));
      if (tuple.some((num) => Number.isNaN(num))) {
        errors.push(`Line ${index + 1}: unable to parse values.`);
        return;
      }
      coords.push(tuple as number[]);
    });

  return { coords, errors };
};

const sanitizeValue = (value: unknown): unknown => {
  if (Array.isArray(value)) {
    const cleaned = value
      .map(sanitizeValue)
      .filter((item) => item !== undefined && item !== null);
    return cleaned.length ? cleaned : undefined;
  }

  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
      .map(([key, val]) => [key, sanitizeValue(val)] as const)
      .filter(([, val]) => {
        if (val === undefined || val === null) {
          return false;
        }
        if (Array.isArray(val)) {
          return val.length > 0;
        }
        if (typeof val === 'object') {
          return Object.keys(val as Record<string, unknown>).length > 0;
        }
        return true;
      });
    if (!entries.length) {
      return undefined;
    }
    return Object.fromEntries(entries);
  }

  if (value === '' || value === undefined) {
    return undefined;
  }

  return value;
};

const ConfigBuilder = () => {
  const initialState = useMemo(() => {
    const defaults = deriveDefaults(builderKeys);
    if (!defaults.outputs) {
      defaults.outputs = ['region_labels'];
    }
    if (!defaults.atlas_names) {
      defaults.atlas_names = ['harvard-oxford', 'juelich'];
    }
    defaults.sources = Array.isArray((defaults as any).sources)
      ? (defaults as any).sources
      : [];
    defaults.data_dir = typeof defaults.data_dir === 'string' ? defaults.data_dir : '';
    defaults.prompt_type = typeof (defaults as any).prompt_type === 'string'
      ? (defaults as any).prompt_type
      : 'summary';
    if (defaults.prompt_type !== 'custom') {
      defaults.custom_prompt = null;
    } else if (typeof defaults.custom_prompt !== 'string') {
      defaults.custom_prompt = '';
    }

    const inferredInputType: InputMode = (() => {
      if (typeof defaults.input_type === 'string') {
        if ((defaults.input_type as string).toLowerCase() === 'region_names') {
          return 'region_names';
        }
      }
      if (defaultRegionNames.length > 0) {
        return 'region_names';
      }
      return 'coords';
    })();

    defaults.input_type = inferredInputType;

    return {
      defaults,
      inputMode: inferredInputType,
      regionNamesText: defaultRegionNames.join('\n')
    };
  }, []);

  const [inputMode, setInputMode] = useState<InputMode>(initialState.inputMode);
  const [coordEntryMode, setCoordEntryMode] = useState<CoordEntryMode>('paste');
  const [coordinateText, setCoordinateText] = useState('30, -22, 50');
  const [coordsFile, setCoordsFile] = useState('');
  const [regionNamesText, setRegionNamesText] = useState(initialState.regionNamesText);
  const [enableStudy, setEnableStudy] = useState(false);
  const [enableSummary, setEnableSummary] = useState(true);
  const [yamlCopied, setYamlCopied] = useState<'idle' | 'copied' | 'error'>('idle');
  const [cliCopied, setCliCopied] = useState<'idle' | 'copied' | 'error'>('idle');

  const [formData, setFormData] = useState<FormState>(() => initialState.defaults);
  const promptType = typeof formData.prompt_type === 'string' && formData.prompt_type
    ? (formData.prompt_type as string)
    : 'summary';

  const { coords, errors: coordErrors } = useMemo(
    () => parseCoordinateText(coordinateText),
    [coordinateText]
  );

  const regionNameList = useMemo(
    () =>
      regionNamesText
        .split(/\r?\n+/)
        .map((name) => name.trim())
        .filter(Boolean),
    [regionNamesText]
  );

  const uiSchema: UiSchema = useMemo(
    () => ({
      'ui:order': builderKeys,
      data_dir: {
        'ui:widget': 'text',
        'ui:emptyValue': null,
        'ui:placeholder': '/path/to/data-directory'
      },
      sources: {
        'ui:widget': 'checkboxes'
      },
      atlas_names: {
        'ui:widget': 'atlasMultiSelect'
      },
      output_format: {
        'ui:field': 'outputFormatField',
        'ui:emptyValue': null
      },
      outputs: {
        'ui:widget': 'checkboxes'
      },
      studies: enableStudy ? {} : { 'ui:widget': 'hidden' },
      study_sources: enableStudy ? {} : { 'ui:widget': 'hidden' },
      study_radius: enableStudy ? {} : { 'ui:widget': 'hidden' },
      study_limit: enableStudy ? {} : { 'ui:widget': 'hidden' },
      prompt_type: enableSummary
        ? { 'ui:field': 'promptTypeField' }
        : { 'ui:widget': 'hidden' },
      summary_model: enableSummary
        ? { 'ui:field': 'summaryModelField' }
        : { 'ui:widget': 'hidden' },
      summary_max_tokens: enableSummary ? {} : { 'ui:widget': 'hidden' },
      custom_prompt: enableSummary
        ? { 'ui:field': 'customPromptField' }
        : { 'ui:widget': 'hidden' },
      input_type: { 'ui:widget': 'hidden' }
    }),
    [enableStudy, enableSummary, promptType]
  );

  const handleInputModeChange = (nextMode: InputMode) => {
    setInputMode(nextMode);
    setFormData((current) => ({
      ...current,
      input_type: nextMode
    }));
  };

  const handleCoordEntryModeChange = (nextMode: CoordEntryMode) => {
    setCoordEntryMode(nextMode);
  };

  const handleRegionNamesInput = (value: string) => {
    setRegionNamesText(value);
  };

  const handleFormChange = useCallback(
    (event: IChangeEvent<FormState>) => {
      const next = { ...(event.formData as FormState) };
      let normalizedMode: InputMode = inputMode;
      if (typeof next.input_type === 'string') {
        normalizedMode = next.input_type === 'region_names' ? 'region_names' : 'coords';
        if (normalizedMode !== inputMode) {
          setInputMode(normalizedMode);
        }
      }
      const normalizedPromptType =
        typeof next.prompt_type === 'string' && next.prompt_type
          ? (next.prompt_type as string)
          : 'summary';
      if (normalizedPromptType !== 'custom') {
        next.custom_prompt = null;
      }
      setFormData({
        ...next,
        input_type: normalizedMode,
        prompt_type: normalizedPromptType
      });
    },
    [inputMode]
  );

  const toggleStudy = () => {
    setEnableStudy((prev) => {
      const next = !prev;
      setFormData((current) => {
        const updated = { ...current };
        if (!next) {
          updated.studies = null;
          updated.study_sources = null;
          updated.study_radius = null;
          updated.study_limit = null;
        } else {
          updated.studies = updated.studies ?? [];
        }
        return updated;
      });
      return next;
    });
  };

  const toggleSummary = () => {
    setEnableSummary((prev) => {
      const next = !prev;
      setFormData((current) => {
        const updated = { ...current };
        if (!next) {
          updated.prompt_type = null;
          updated.summary_model = null;
          updated.summary_max_tokens = null;
          updated.custom_prompt = null;
        } else {
          if (typeof updated.prompt_type !== 'string' || !updated.prompt_type) {
            updated.prompt_type = 'summary';
          }
          if (updated.prompt_type !== 'custom') {
            updated.custom_prompt = null;
          }
        }
        return updated;
      });
      return next;
    });
  };

  const configData = useMemo(() => {
    const payload: Record<string, unknown> = {
      ...formData
    };

    if (inputMode === 'coords') {
      payload.input_type = 'coords';
      payload.coordinates = coordEntryMode === 'paste' ? (coords.length ? coords : null) : null;
      payload.coords_file = coordEntryMode === 'file' ? (coordsFile || null) : null;
      payload.region_names = null;
    } else {
      payload.input_type = 'region_names';
      payload.coordinates = null;
      payload.coords_file = null;
      payload.region_names = regionNameList.length ? regionNameList : null;
    }

    if (!enableStudy) {
      payload.studies = null;
      payload.study_sources = null;
      payload.study_radius = null;
      payload.study_limit = null;
    }

    if (!enableSummary) {
      payload.prompt_type = null;
      payload.summary_model = null;
      payload.summary_max_tokens = null;
      payload.custom_prompt = null;
    } else if (payload.prompt_type !== 'custom') {
      payload.custom_prompt = null;
    }

    const atlasConfigs = deriveAtlasConfigs(formData.atlas_names);
    if (Object.keys(atlasConfigs).length > 0) {
      payload.atlas_configs = atlasConfigs;
    } else {
      delete payload.atlas_configs;
    }

    return (sanitizeValue(payload) as Record<string, unknown>) ?? {};
  }, [
    formData,
    inputMode,
    coordEntryMode,
    coords,
    coordsFile,
    regionNameList,
    enableStudy,
    enableSummary
  ]);

  const yamlPreview = useMemo(() => {
    try {
      return YAML.dump(configData, { lineWidth: 120 });
    } catch (error) {
      console.error('Unable to render YAML preview', error);
      return '# Unable to render YAML preview';
    }
  }, [configData]);

  const cliCommand = 'coord2region --config coord2region-config.yaml';

  const copyToClipboard = useCallback(async (value: string, onComplete: (state: 'idle' | 'copied' | 'error') => void) => {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(value);
      } else {
        const textarea = document.createElement('textarea');
        textarea.value = value;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
      }
      onComplete('copied');
      setTimeout(() => onComplete('idle'), 2000);
    } catch (error) {
      console.error('Clipboard copy failed', error);
      onComplete('error');
      setTimeout(() => onComplete('idle'), 2500);
    }
  }, []);

  const handleDownload = () => {
    const blob = new Blob([yamlPreview], { type: 'text/yaml' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'coord2region-config.yaml';
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="config-builder">
      <div className="config-section">
        <h4>Input Type</h4>
        <div className="mode-toggle" role="radiogroup" aria-label="Select input type">
          <button
            type="button"
            className={clsx('toggle', inputMode === 'coords' && 'toggle--active')}
            onClick={() => handleInputModeChange('coords')}
            aria-checked={inputMode === 'coords'}
            role="radio"
          >
            Coordinates
          </button>
          <button
            type="button"
            className={clsx('toggle', inputMode === 'region_names' && 'toggle--active')}
            onClick={() => handleInputModeChange('region_names')}
            aria-checked={inputMode === 'region_names'}
            role="radio"
          >
            Region names
          </button>
        </div>

        {inputMode === 'coords' ? (
          <>
            <div className="mode-toggle" role="radiogroup" aria-label="Coordinate input mode">
              <button
                type="button"
                className={clsx('toggle', coordEntryMode === 'paste' && 'toggle--active')}
                onClick={() => handleCoordEntryModeChange('paste')}
                aria-checked={coordEntryMode === 'paste'}
                role="radio"
              >
                Paste coordinates
              </button>
              <button
                type="button"
                className={clsx('toggle', coordEntryMode === 'file' && 'toggle--active')}
                onClick={() => handleCoordEntryModeChange('file')}
                aria-checked={coordEntryMode === 'file'}
                role="radio"
              >
                Use coordinate file
              </button>
            </div>

            {coordEntryMode === 'paste' ? (
              <div className="card">
                <label htmlFor="coord-textarea" className="field-label tooltip" data-tooltip={tooltipFromSchema('coordinates')}>
                  {schema.properties?.coordinates && typeof schema.properties.coordinates === 'object'
                    ? (schema.properties.coordinates as SchemaProperty).title || 'Coordinates'
                    : 'Coordinates'}
                </label>
                <textarea
                  id="coord-textarea"
                  value={coordinateText}
                  onChange={(event) => setCoordinateText(event.target.value)}
                  placeholder="30, -22, 50"
                  rows={6}
                />
                {coordErrors.length > 0 ? (
                  <ul className="form-errors">
                    {coordErrors.map((message) => (
                      <li key={message}>{message}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="helper">Parsed {coords.length} coordinate triplet{coords.length === 1 ? '' : 's'}.</p>
                )}
              </div>
            ) : (
              <div className="card">
                <label htmlFor="coord-file" className="field-label tooltip" data-tooltip={tooltipFromSchema('coords_file')}>
                  Coordinate file path
                </label>
                <input
                  id="coord-file"
                  type="text"
                  value={coordsFile}
                  onChange={(event) => setCoordsFile(event.target.value)}
                  placeholder="/path/to/coordinates.tsv"
                />
                <p className="helper">Provide a local path to a CSV/TSV/XLSX file.</p>
              </div>
            )}
          </>
        ) : (
          <div className="card">
            <label htmlFor="region-names-textarea" className="field-label tooltip" data-tooltip={tooltipFromSchema('region_names')}>
              Region names
            </label>
            <textarea
              id="region-names-textarea"
              value={regionNamesText}
              onChange={(event) => handleRegionNamesInput(event.target.value)}
              placeholder="Amygdala\nHippocampus"
              rows={6}
            />
            <p className="helper">
              Enter one region per line. Parsed {regionNameList.length} region name{regionNameList.length === 1 ? '' : 's'}.
            </p>
          </div>
        )}
      </div>

      <div className="config-section">
        <div className="card card--inline">
          <div>
            <h4>Study review</h4>
            <p>Control which papers to inspect when running region lookups.</p>
          </div>
          <button type="button" className={clsx('toggle', enableStudy && 'toggle--active')} onClick={toggleStudy}>
            {enableStudy ? 'Enabled' : 'Disabled'}
          </button>
        </div>
        <div className="card card--inline">
          <div>
            <h4>Summaries</h4>
            <p>Generate natural language outputs via configured providers.</p>
          </div>
          <button type="button" className={clsx('toggle', enableSummary && 'toggle--active')} onClick={toggleSummary}>
            {enableSummary ? 'Enabled' : 'Disabled'}
          </button>
        </div>
      </div>

      <div className="config-section">
        <div className="card">
          <Form
            schema={builderSchema}
            formData={formData}
            onChange={handleFormChange}
            validator={validator}
            uiSchema={uiSchema}
            widgets={widgets}
            fields={fields}
            FieldTemplate={FieldTemplate}
            formContext={{ promptType }}
            liveValidate={false}
            noHtml5Validate
          >
            <div className="form-footer">
              <small>Updates apply immediately to the YAML preview.</small>
            </div>
          </Form>
        </div>
      </div>

      <aside className="config-preview">
        <div className="card">
          <div className="preview-header">
            <h4>YAML preview</h4>
            <div className="config-actions">
              <button
                type="button"
                onClick={() => copyToClipboard(yamlPreview, setYamlCopied)}
              >
                Copy YAML
              </button>
              <button type="button" onClick={handleDownload}>Download YAML</button>
            </div>
          </div>
          <pre className="yaml-output" aria-live="polite">
            <code>{yamlPreview}</code>
          </pre>
          {yamlCopied === 'copied' && <p className="status status--success">YAML copied to clipboard.</p>}
          {yamlCopied === 'error' && <p className="status status--error">Unable to copy YAML automatically.</p>}
        </div>
        <div className="card">
          <div className="preview-header">
            <h4>CLI command</h4>
            <div className="config-actions">
              <button
                type="button"
                onClick={() => copyToClipboard(cliCommand, setCliCopied)}
              >
                Copy command
              </button>
            </div>
          </div>
          <code className="cli-command">{cliCommand}</code>
          {cliCopied === 'copied' && <p className="status status--success">Command copied.</p>}
          {cliCopied === 'error' && <p className="status status--error">Unable to copy command.</p>}
        </div>
      </aside>
    </section>
  );
};

export default ConfigBuilder;
