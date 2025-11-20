import { KeyboardEvent, useCallback, useEffect, useMemo, useState, ChangeEvent } from 'react';
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

type ConfigBuilderProps = {
  showHeaderNav?: boolean;
};

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

// Friendly display names for atlas values
const atlasDisplayNames: Record<string, string> = {
  // Volumetric (nilearn)
  'aal': 'AAL',
  'basc': 'BASC',
  'brodmann': 'Brodmann',
  'destrieux': 'Destrieux',
  'harvard-oxford': 'Harvard–Oxford',
  'juelich': 'Jülich',
  'pauli': 'Pauli',
  'schaefer': 'Schaefer',
  'talairach': 'Talairach',
  'yeo': 'Yeo',
  // Surface (mne)
  'aparc': 'aparc (Desikan–Killiany)',
  'aparc.a2005s': 'aparc a2005s',
  'aparc.a2009s': 'aparc a2009s',
  'aparc_sub': 'aparc_sub',
  'human-connectum project': 'Human Connectome Project',
  'oasis.chubs': 'OASIS CHUBS',
  'pals_b12_lobes': 'PALS-B12 Lobes',
  'pals_b12_orbitofrontal': 'PALS-B12 Orbitofrontal',
  'pals_b12_visuotopic': 'PALS-B12 Visuotopic',
  'yeo2011': 'Yeo (2011)',
  // Coordinates (mne)
  'dosenbach': 'Dosenbach',
  'power': 'Power',
  'seitzman': 'Seitzman'
};

const humanizeAtlas = (value: string): string => {
  // Fallback humanization: replace separators with spaces and capitalize words
  return value
    .replace(/[._-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (m) => m.toUpperCase());
};

const displayAtlas = (value: string): string => atlasDisplayNames[value] || humanizeAtlas(value);

const deepClone = <T,>(value: T): T => JSON.parse(JSON.stringify(value));

const datasetSourceOptions = ["neurosynth", "neuroquery", "nidm_pain"] as const;
const outputFormatOptions = ["json", "pickle", "csv", "pdf", "directory"] as const;
type SelectOption = { value: string; label: string };

const summaryModelOptions: ReadonlyArray<SelectOption> = [
  { value: 'gpt-4o-mini', label: 'GPT-4o Mini (OpenAI)' },
  { value: 'gpt-4o', label: 'GPT-4o (OpenAI)' },
  { value: 'o4-mini', label: 'o4-mini (OpenAI)' },
  { value: 'o4', label: 'o4 (OpenAI)' },
  { value: 'gpt-4.1-mini', label: 'GPT-4.1 Mini (OpenAI)' },
  { value: 'gpt-4.1', label: 'GPT-4.1 (OpenAI)' },
  { value: 'o3-mini', label: 'o3-mini (OpenAI)' },
  { value: 'gemini-2.0-flash', label: 'Gemini 2.0 Flash (Google)' },
  { value: 'gemini-1.5-pro', label: 'Gemini 1.5 Pro (Google)' },
  { value: 'gemini-1.0-pro', label: 'Gemini 1.0 Pro (Google)' },
  { value: 'claude-3-haiku', label: 'Claude 3 Haiku (Anthropic)' },
  { value: 'claude-3-opus', label: 'Claude 3 Opus (Anthropic)' },
  { value: 'deepseek-r1', label: 'DeepSeek R1 (OpenRouter)' },
  { value: 'deepseek-chat-v3-0324', label: 'DeepSeek Chat v3 (OpenRouter)' },
  { value: 'distilgpt2', label: 'distilgpt2 (Hugging Face)' }
];

// Image prompt type options
const imagePromptTypeOptions: ReadonlyArray<SelectOption> = [
  { value: 'anatomical', label: 'Anatomical' },
  { value: 'functional', label: 'Functional' },
  { value: 'schematic', label: 'Schematic' },
  { value: 'artistic', label: 'Artistic' },
  { value: 'custom', label: 'Custom prompt' }
];

// Map models to their API key providers
const modelToProvider: Record<string, string> = {
  'gpt-4o-mini': 'openai',
  'gpt-4o': 'openai',
  'o4-mini': 'openai',
  'o4': 'openai',
  'gpt-4.1-mini': 'openai',
  'gpt-4.1': 'openai',
  'o3-mini': 'openai',
  'gemini-2.0-flash': 'gemini',
  'gemini-1.5-pro': 'gemini',
  'gemini-1.0-pro': 'gemini',
  'claude-3-haiku': 'anthropic',
  'claude-3-opus': 'anthropic',
  'deepseek-r1': 'openrouter',
  'deepseek-chat-v3-0324': 'openrouter',
  'distilgpt2': 'huggingface'
};

// Provider display names
const providerDisplayNames: Record<string, string> = {
  'gemini': 'Google Gemini',
  'anthropic': 'Anthropic Claude',
  'openrouter': 'OpenRouter',
  'openai': 'OpenAI',
  'huggingface': 'Hugging Face'
};

const promptTypeOptions: ReadonlyArray<SelectOption> = [
  { value: 'summary', label: 'Integrated summary' },
  { value: 'region_name', label: 'Region name focus' },
  { value: 'function', label: 'Functional profile' },
  { value: 'custom', label: 'Custom prompt' }
];

// Suggested image model options (single selection)
const imageModelOptions: ReadonlyArray<SelectOption> = [
  { value: 'stabilityai/stable-diffusion-2', label: 'Stable Diffusion 2 (Stability)' },
  { value: 'runwayml/stable-diffusion-v1-5', label: 'Stable Diffusion v1-5 (RunwayML)' },
  { value: 'stabilityai/sd-turbo', label: 'SD Turbo (Stability)' },
  { value: 'stabilityai/stable-diffusion-xl-base-1.0', label: 'Stable Diffusion XL Base 1.0' },
  { value: 'gpt-image-1', label: 'GPT-Image-1 (OpenAI)' }
];

// Synthetic properties for image generation controls (not present in exported schema)
const imageBackendProperty: RJSFSchema = {
  type: 'string',
  enum: ['ai', 'nilearn', 'both'],
  default: 'ai',
  title: 'Image backend'
};

const imageModelProperty: RJSFSchema = {
  type: 'string',
  default: 'stabilityai/stable-diffusion-2',
  title: 'Image model'
};

const imagePromptTypeProperty: RJSFSchema = {
  type: 'string',
  enum: imagePromptTypeOptions.map((o) => o.value),
  default: 'anatomical',
  title: 'Image prompt type'
};

const imageCustomPromptProperty: RJSFSchema = {
  type: 'string',
  default: '',
  title: 'Custom image prompt'
};


const atlasProperty = (() => {
  const property = schema.properties?.atlas_names;
  if (!property || typeof property !== 'object') {
    return undefined;
  }
  const cloned = deepClone(property);
  if (Array.isArray((cloned as RJSFSchema).anyOf)) {
    const arrayOption = (cloned as RJSFSchema).anyOf?.find((option: any) => option?.type === 'array');
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
    const arrayOption = cloned.anyOf.find((option: any) => option?.type === 'array');
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

const workingDirectoryProperty = (() => {
  const property = schema.properties?.working_directory;
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

const outputNameProperty = (() => {
  const property = schema.properties?.output_name;
  if (!property || typeof property !== 'object') {
    return undefined;
  }
  const cloned = deepClone(property) as RJSFSchema;
  delete cloned.anyOf;
  cloned.type = 'string';
  if (cloned.default === null || cloned.default === undefined) {
    cloned.default = '';
  }
  return cloned;
})();

const customPromptProperty = (() => {
  const property = schema.properties?.custom_prompt;
  if (!property || typeof property !== 'object') {
    return undefined;
  }
  const cloned = deepClone(property) as RJSFSchema;
  delete cloned.anyOf;
  cloned.type = 'string';
  if (cloned.default === null || cloned.default === undefined) {
    cloned.default = '';
  }
  return cloned;
})();

const summaryModelsProperty = (() => {
  const property = schema.properties?.summary_models;
  if (!property || typeof property !== 'object') {
    return undefined;
  }
  const cloned = deepClone(property) as RJSFSchema;
  delete cloned.anyOf;
  cloned.type = 'array';
  cloned.items = { type: 'string' };
  if (cloned.default === null || cloned.default === undefined) {
    cloned.default = [];
  }
  return cloned;
})();

const summaryMaxTokensProperty = (() => {
  const property = schema.properties?.summary_max_tokens;
  if (!property || typeof property !== 'object') {
    return undefined;
  }
  const cloned = deepClone(property) as RJSFSchema;
  delete cloned.anyOf;
  cloned.type = 'string';
  if (cloned.default === null || cloned.default === undefined) {
    cloned.default = '';
  }
  return cloned;
})();

const emailForAbstractsProperty = (() => {
  const property = schema.properties?.email_for_abstracts;
  if (!property || typeof property !== 'object') {
    return undefined;
  }
  const cloned = deepClone(property) as RJSFSchema;
  delete cloned.anyOf;
  cloned.type = 'string';
  if (cloned.default === null || cloned.default === undefined) {
    cloned.default = '';
  }
  return cloned;
})();

// Flatten API key fields to simple strings to avoid RJSF anyOf ("Option 1/2") selectors
const anthropicApiKeyProperty = (() => {
  const property = schema.properties?.anthropic_api_key;
  if (!property || typeof property !== 'object') {
    return undefined;
  }
  const cloned = deepClone(property) as RJSFSchema;
  delete cloned.anyOf;
  cloned.type = 'string';
  if (cloned.default === null || cloned.default === undefined) {
    cloned.default = '';
  }
  return cloned;
})();

const openaiApiKeyProperty = (() => {
  const property = schema.properties?.openai_api_key;
  if (!property || typeof property !== 'object') {
    return undefined;
  }
  const cloned = deepClone(property) as RJSFSchema;
  delete cloned.anyOf;
  cloned.type = 'string';
  if (cloned.default === null || cloned.default === undefined) {
    cloned.default = '';
  }
  return cloned;
})();

const openrouterApiKeyProperty = (() => {
  const property = schema.properties?.openrouter_api_key;
  if (!property || typeof property !== 'object') {
    return undefined;
  }
  const cloned = deepClone(property) as RJSFSchema;
  delete cloned.anyOf;
  cloned.type = 'string';
  if (cloned.default === null || cloned.default === undefined) {
    cloned.default = '';
  }
  return cloned;
})();

const geminiApiKeyProperty = (() => {
  const property = schema.properties?.gemini_api_key;
  if (!property || typeof property !== 'object') {
    return undefined;
  }
  const cloned = deepClone(property) as RJSFSchema;
  delete cloned.anyOf;
  cloned.type = 'string';
  if (cloned.default === null || cloned.default === undefined) {
    cloned.default = '';
  }
  return cloned;
})();

const huggingfaceApiKeyProperty = (() => {
  const property = schema.properties?.huggingface_api_key;
  if (!property || typeof property !== 'object') {
    return undefined;
  }
  const cloned = deepClone(property) as RJSFSchema;
  delete cloned.anyOf;
  cloned.type = 'string';
  if (cloned.default === null || cloned.default === undefined) {
    cloned.default = '';
  }
  return cloned;
})();

const builderKeys: string[] = [
  'input_type',
  'working_directory',
  'sources',
  'atlas_names',
  'outputs',
  'image_backend',
  'image_model',
  'image_prompt_type',
  'image_custom_prompt',
  'output_format',
  'output_name',
  'study_search_radius',
  'region_search_radius',
  'prompt_type',
  'custom_prompt',
  'summary_models',
  'summary_max_tokens',
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
    ? schema.required.filter((key: string) => builderKeys.includes(key))
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

if (builderSchema.properties?.working_directory && workingDirectoryProperty) {
  builderSchema.properties.working_directory = workingDirectoryProperty;
}

if (builderSchema.properties?.output_name && outputNameProperty) {
  builderSchema.properties.output_name = outputNameProperty;
}

if (builderSchema.properties?.custom_prompt && customPromptProperty) {
  builderSchema.properties.custom_prompt = customPromptProperty;
}

if (builderSchema.properties?.summary_models && summaryModelsProperty) {
  builderSchema.properties.summary_models = summaryModelsProperty;
}

if (builderSchema.properties?.summary_max_tokens && summaryMaxTokensProperty) {
    builderSchema.properties.summary_max_tokens = summaryMaxTokensProperty;
}
if (builderSchema.properties?.email_for_abstracts && emailForAbstractsProperty) {
  builderSchema.properties.email_for_abstracts = emailForAbstractsProperty;
}

// Apply flattened API key properties
if (builderSchema.properties?.anthropic_api_key && anthropicApiKeyProperty) {
  builderSchema.properties.anthropic_api_key = anthropicApiKeyProperty;
}
if (builderSchema.properties?.openai_api_key && openaiApiKeyProperty) {
  builderSchema.properties.openai_api_key = openaiApiKeyProperty;
}
if (builderSchema.properties?.openrouter_api_key && openrouterApiKeyProperty) {
  builderSchema.properties.openrouter_api_key = openrouterApiKeyProperty;
}
if (builderSchema.properties?.gemini_api_key && geminiApiKeyProperty) {
  builderSchema.properties.gemini_api_key = geminiApiKeyProperty;
}
if (builderSchema.properties?.huggingface_api_key && huggingfaceApiKeyProperty) {
  builderSchema.properties.huggingface_api_key = huggingfaceApiKeyProperty;
}

// Inject synthetic image properties (override to flatten anyOf and avoid Option 1/2)
builderSchema.properties = builderSchema.properties || {};
builderSchema.properties.image_backend = imageBackendProperty;
builderSchema.properties.image_model = imageModelProperty;
builderSchema.properties.image_prompt_type = imagePromptTypeProperty;
builderSchema.properties.image_custom_prompt = imageCustomPromptProperty;

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
  const { id, classNames, label, required, description, errors, help, children, hidden, uiSchema } = props as FieldTemplateProps & { uiSchema?: any };
  const displayLabel = (props as any).displayLabel !== false;
  if (hidden) {
    return (
      <div className="form-field" style={{ display: 'none' }}>
        {children}
      </div>
    );
  }
  const key = id.replace(/^root_/, '').split('_')[0];
  const tooltip = tooltipMap[key];
  const labelText = (!displayLabel || (uiSchema && uiSchema['ui:options'] && uiSchema['ui:options'].label === false))
    ? ''
    : (label || key);

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

// Suppress default schema title/description that RJSF renders at the root
const TitleFieldTemplate = () => null;
const DescriptionFieldTemplate = () => null;

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

type AtlasSelectionProps = {
  selected: string[];
  onChange: (next: string[]) => void;
  enforceSingle?: boolean; // when true, allow selecting only one atlas at a time
};

const AtlasSelection = ({ selected, onChange, enforceSingle = false }: AtlasSelectionProps) => {
  const [open, setOpen] = useState<Record<string, boolean>>({
    'volumetric-nilearn': true,
    'surface-mne': false,
    'coordinates-mne': false
  });
  const [filters, setFilters] = useState<Record<string, string>>({});

  const toggleOpen = (id: string) => setOpen((s) => ({ ...s, [id]: !s[id] }));

  const commit = (next: Iterable<string>) => {
    let unique = Array.from(new Set(Array.from(next))).sort((a, b) => a.localeCompare(b));
    if (enforceSingle && unique.length > 1) {
      // Keep only the last chosen item when single selection is enforced
      unique = [unique[unique.length - 1]];
    }
    onChange(unique);
  };

  const toggleAtlas = (name: string) => {
    const set = new Set(selected);
    if (enforceSingle) {
      if (set.has(name)) {
        set.delete(name);
        commit(set);
      } else {
        commit([name]);
      }
    } else {
      set.has(name) ? set.delete(name) : set.add(name);
      commit(set);
    }
  };

  const handleGroupToggle = (options: string[]) => {
    if (enforceSingle) return; // Disable group toggle in single-select mode
    const set = new Set(selected);
    const hasMissing = options.some((opt) => !set.has(opt));
    options.forEach((opt) => (hasMissing ? set.add(opt) : set.delete(opt)));
    commit(set);
  };

  const groups = atlasGroups;

  return (
    <div className="atlas-select">
      {groups.map((group) => {
        const query = (filters[group.id] || '').toLowerCase();
        const filtered = query
          ? group.options.filter((opt) => opt.toLowerCase().includes(query))
          : group.options;
        const groupSelected = filtered.filter((opt) => selected.includes(opt));
        const allSelected = filtered.length > 0 && groupSelected.length === filtered.length;
        return (
          <details key={group.id} className="atlas-select__group" open={!!open[group.id]}>
            <summary className="atlas-select__summary" onClick={(e) => { e.preventDefault(); toggleOpen(group.id); }}>
              <span className="atlas-select__chevron" aria-hidden="true" />
              <span className="atlas-select__title">{group.label}</span>
              <span className="atlas-select__meta" aria-label="selected count">{groupSelected.length}/{filtered.length || 0}</span>
              <button
                type="button"
                className="atlas-select__toggle"
                disabled={enforceSingle}
                onClick={(e) => { e.preventDefault(); handleGroupToggle(filtered); }}
              >
                {allSelected ? `Clear all (${filtered.length})` : `Select all (${filtered.length})`}
              </button>
            </summary>
            <div className="atlas-select__body">
              <div className="atlas-select__toolbar">
                <input
                  type="text"
                  className="atlas-select__search"
                  placeholder="Filter atlas names..."
                  value={filters[group.id] || ''}
                  onChange={(e) => setFilters((m) => ({ ...m, [group.id]: e.target.value }))}
                />
              </div>
              {filtered.length === 0 ? (
                <p className="atlas-select__empty">No matching atlas.</p>
              ) : (
                <ul className="atlas-select__list">
                  {filtered.map((opt) => (
                    <li key={`${group.id}-${opt}`} className={clsx('atlas-select__item', selected.includes(opt) && 'is-selected')}>
                      <label>
                        <input
                          type="checkbox"
                          checked={selected.includes(opt)}
                          onChange={() => toggleAtlas(opt)}
                        />
                        <span className="atlas-select__name">{displayAtlas(opt)}</span>
                      </label>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </details>
        );
      })}
    </div>
  );
};

const SummaryModelMultiSelect = ({ formData, onChange, idSchema }: FieldProps) => {
  const value = Array.isArray(formData) ? formData : [];
  const inputId = idSchema?.$id ?? 'summary-models';
  const [inputValue, setInputValue] = useState('');
  
  // Calculate required providers based on selected models
  const requiredProviders = Array.from(new Set(
    value.map(model => modelToProvider[model]).filter(Boolean)
  ));

  const handleAddModel = (modelValue: string) => {
    const trimmedValue = modelValue.trim();
    if (trimmedValue && !value.includes(trimmedValue)) {
      onChange([...value, trimmedValue]);
    }
    setInputValue('');
  };

  const handleRemoveModel = (modelToRemove: string) => {
    onChange(value.filter(model => model !== modelToRemove));
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      handleAddModel(inputValue);
    }
  };

  return (
    <div className="form-field">
      <label className="field-label" htmlFor={inputId}>Summary Models</label>
      
      {/* Selected models display */}
      {value.length > 0 && (
        <div className="selected-items">
          {value.map((model) => (
            <span key={model} className="selected-item">
              {model}
              <button
                type="button"
                className="remove-item"
                onClick={() => handleRemoveModel(model)}
                aria-label={`Remove ${model}`}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}

      {/* Add new model input */}
      <input
        type="text"
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        onKeyPress={handleKeyPress}
        placeholder="Type model name and press Enter to add"
        list={`${inputId}-options`}
      />
      <datalist id={`${inputId}-options`}>
        {summaryModelOptions.map((option) => (
          <option key={option.value} value={option.value}>{option.label}</option>
        ))}
      </datalist>
      
      <p className="helper">
        Type model names and press Enter to add them. You can use any model identifier supported by your providers.
      </p>
      
      {/* Show required API keys */}
      {value.length > 0 && (
        <div className="api-key-requirements">
          <p className="helper">
            <strong>Required API Keys:</strong> {requiredProviders.map(provider => providerDisplayNames[provider]).join(', ')}
          </p>
        </div>
      )}
    </div>
  );
};

const ApiKeyField = ({ formData, onChange, idSchema, uiSchema }: FieldProps) => {
  const value = typeof formData === 'string' ? formData : '';
  const inputId = idSchema?.$id ?? 'api-key';
  const placeholder = uiSchema?.['ui:placeholder'] || 'Enter API key';
  
  return (
    <div className="form-field">
      <label className="field-label" htmlFor={inputId}>
        {uiSchema?.['ui:title'] || 'API Key'}
      </label>
      <input
        id={inputId}
        type="text"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        className="api-key-input"
      />
      <p className="helper">
        Enter your API key for this provider.
      </p>
    </div>
  );
};

const ImageModelField = ({ formData, onChange, idSchema }: FieldProps) => {
  // Single-value selector inspired by SummaryModelMultiSelect
  const value = typeof formData === 'string' ? formData : '';
  const inputId = idSchema?.$id ?? 'image-model';
  const datalistId = `${inputId}-options`;
  const [inputValue, setInputValue] = useState('');

  const setModel = (model: string) => {
    const trimmed = (model || '').trim();
    onChange(trimmed);
    setInputValue('');
  };

  const clearModel = () => {
    onChange('');
    setInputValue('');
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      setModel(inputValue);
    }
  };

  return (
    <div className="form-field">
      <label className="field-label" htmlFor={inputId}>Image Model</label>

      {/* Selected model chip */}
      {value && (
        <div className="selected-items" style={{ marginBottom: 6 }}>
          <span className="selected-item">
            {value}
            <button
              type="button"
              className="remove-item"
              onClick={clearModel}
              aria-label={`Remove ${value}`}
            >
              ×
            </button>
          </span>
        </div>
      )}

      {/* Input with datalist suggestions */}
      <input
        id={inputId}
        type="text"
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        onKeyPress={handleKeyPress}
        placeholder={value ? 'Type to change model, press Enter' : 'stabilityai/stable-diffusion-2'}
        list={datalistId}
      />
      <datalist id={datalistId}>
        {imageModelOptions.map((option) => (
          <option key={option.value} value={option.value}>{option.label}</option>
        ))}
      </datalist>

      <p className="helper">
        Type a model identifier and press Enter, or pick from the suggestions.
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
        Select a template for generated summaries. Choose "Custom prompt" to provide your own wording below.
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
        Use {'{coord}'} for the coordinate. This placeholder is filled automatically before calling the model.
      </p>
    </div>
  );
};

const widgets = {
  atlasMultiSelect: AtlasMultiSelect
};

const fields = {
  summaryModelMultiSelect: SummaryModelMultiSelect,
  promptTypeField: PromptTypeField,
  customPromptField: CustomPromptField,
  outputFormatField: OutputFormatField,
  apiKeyField: ApiKeyField,
  imageModelField: ImageModelField
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

const ConfigBuilder = (props: ConfigBuilderProps = {}) => {
  const initialState = useMemo(() => {
    const defaults = deriveDefaults(builderKeys);
    if (!defaults.outputs) {
      defaults.outputs = ['region_labels'];
    }
    if (!defaults.atlas_names) {
      defaults.atlas_names = ['harvard-oxford', 'juelich'];
    }
    // Image generation defaults
    (defaults as any).image_backend = typeof (defaults as any).image_backend === 'string' && (defaults as any).image_backend 
      ? (defaults as any).image_backend 
      : 'both'; // Changed default to 'both'
    (defaults as any).image_model = typeof (defaults as any).image_model === 'string' && (defaults as any).image_model 
      ? (defaults as any).image_model 
      : 'runwayml/stable-diffusion-v1-5'; // Changed default model
    defaults.sources = Array.isArray((defaults as any).sources)
      ? (defaults as any).sources
      : [];
    defaults.working_directory = typeof defaults.working_directory === 'string' ? defaults.working_directory : '';
    defaults.output_name = typeof defaults.output_name === 'string' ? defaults.output_name : '';
    defaults.custom_prompt = typeof defaults.custom_prompt === 'string' ? defaults.custom_prompt : '';
    // Start with no summary models by default (since Summaries is off initially)
    defaults.summary_models = Array.isArray(defaults.summary_models)
      ? defaults.summary_models
      : typeof defaults.summary_models === 'string' && defaults.summary_models
        ? [defaults.summary_models]
        : [];
    defaults.summary_max_tokens = typeof defaults.summary_max_tokens === 'number'
      ? defaults.summary_max_tokens.toString()
      : typeof defaults.summary_max_tokens === 'string'
        ? defaults.summary_max_tokens
        : '';
    defaults.region_search_radius = typeof defaults.region_search_radius === 'number'
      ? defaults.region_search_radius.toString()
      : typeof defaults.region_search_radius === 'string'
        ? defaults.region_search_radius
        : '';
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

    // If starting in region_names mode, enforce a single atlas selection
    if (inferredInputType === 'region_names') {
      const currentAtlases = Array.isArray((defaults as any).atlas_names) ? ((defaults as any).atlas_names as string[]) : [];
      (defaults as any).atlas_names = currentAtlases.length > 0 ? [currentAtlases[0]] : ['harvard-oxford'];
    }

    return {
      defaults,
      inputMode: inferredInputType,
      regionNamesText: defaultRegionNames.length > 0
        ? defaultRegionNames.join('\n')
        : 'Amygdala\nHippocampus'
    };
  }, []);

  const [inputMode, setInputMode] = useState<InputMode>(initialState.inputMode);
  const [coordEntryMode, setCoordEntryMode] = useState<CoordEntryMode>('paste');
  const [coordinateText, setCoordinateText] = useState('30, -22, 50');
  const [coordsFile, setCoordsFile] = useState('examples/toy_coordinates.csv');
  const [regionNamesText, setRegionNamesText] = useState(initialState.regionNamesText);
  // Default outputs: Studies enabled by default; Summaries and Images disabled
  const [enableStudy, setEnableStudy] = useState(true);
  const [enableSummary, setEnableSummary] = useState(false);
  const [yamlCopied, setYamlCopied] = useState<'idle' | 'copied' | 'error'>('idle');
  const [cliCopied, setCliCopied] = useState<'idle' | 'copied' | 'error'>('idle');
  const [directCliCopied, setDirectCliCopied] = useState<'idle' | 'copied' | 'error'>('idle');
  const [viewMode, setViewMode] = useState<'builder' | 'about' | 'cloud'>('builder');
  const [outputDetail, setOutputDetail] = useState<'studies' | 'summaries' | 'images'>('studies');
  const [selectedTemplate, setSelectedTemplate] = useState<string>('');
  const [templateUndo, setTemplateUndo] = useState<null | {
    inputMode: InputMode;
    coordEntryMode: CoordEntryMode;
    coordinateText: string;
    coordsFile: string;
    regionNamesText: string;
    enableStudy: boolean;
    enableSummary: boolean;
    formData: any;
  }>(null);
  const [importStatus, setImportStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [importMessage, setImportMessage] = useState<string>('');

  // Helper to clear the 'images' flag from outputs
  const withoutImages = (outputs: unknown) => {
    const arr = Array.isArray(outputs) ? (outputs as string[]) : [];
    return arr.filter((o) => o !== 'images');
  };

  const [formData, setFormData] = useState<FormState>(() => initialState.defaults);
  const promptType = typeof formData.prompt_type === 'string' && formData.prompt_type
    ? (formData.prompt_type as string)
    : 'summary';
  const outputsSelected = Array.isArray(formData.outputs) ? (formData.outputs as string[]) : [];
  const enableImages = outputsSelected.includes('images');
  const imageBackendVal = typeof (formData as any).image_backend === 'string' && (formData as any).image_backend
    ? (formData as any).image_backend as string
    : 'ai';
  const showImageAiOptions = enableImages && (imageBackendVal === 'ai' || imageBackendVal === 'both');

  // Track which section is visible for quick-nav highlighting
  const [activeSection, setActiveSection] = useState<string>('atlas-section');
  useEffect(() => {
    const ids = [
      'atlas-section',
      ...(enableStudy ? ['studies-section'] : []),
      ...(enableSummary ? ['summaries-section'] : []),
      ...(enableImages ? ['images-section'] : []),
      'outputs-section'
    ];
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
        if (visible.length > 0) {
          setActiveSection((visible[0].target as HTMLElement).id);
        }
      },
      {
        root: null,
        rootMargin: '0px 0px -60% 0px',
        threshold: [0.15, 0.35, 0.5, 0.75]
      }
    );
    ids.forEach((id) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });
    return () => observer.disconnect();
  }, [enableStudy, enableSummary, enableImages]);

  // Get required API keys based on selected models
  const selectedModels = Array.isArray(formData.summary_models) 
    ? formData.summary_models as string[]
    : [];
  const requiredProviders = Array.from(new Set(
    selectedModels.map(model => modelToProvider[model]).filter(Boolean)
  ));

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

  // Require coords file path when using file-based coordinate input
  const coordFileRequired = inputMode === 'coords' && coordEntryMode === 'file';
  const coordFileError = useMemo(
    () => coordFileRequired && (!coordsFile || coordsFile.trim().length === 0),
    [coordFileRequired, coordsFile]
  );

  const uiSchema: UiSchema = useMemo(
    () => ({
      'ui:order': builderKeys,
      'ui:title': '',
      'ui:description': '',
      working_directory: { 'ui:widget': 'hidden' },
      output_name: { 'ui:widget': 'hidden' },
      // Hide default RJSF sources; custom Studies section renders sources
      sources: { 'ui:widget': 'hidden' },
      atlas_names: { 'ui:widget': 'hidden' },
      // Hide image fields in the default form; custom Images section renders them
      image_backend: { 'ui:widget': 'hidden' },
      image_model: { 'ui:widget': 'hidden' },
      image_prompt_type: { 'ui:widget': 'hidden' },
      image_custom_prompt: { 'ui:widget': 'hidden' },
      output_format: { 'ui:widget': 'hidden' },
      // Hide outputs entirely; we render a custom Outputs mini-section below
      outputs: { 'ui:widget': 'hidden' },
      // Hide default email; custom Studies section renders it
      email_for_abstracts: { 'ui:widget': 'hidden' },
      // Hide study_search_radius here; custom Studies section renders and manages it
      study_search_radius: { 'ui:widget': 'hidden' },
      region_search_radius: {
        'ui:widget': 'hidden'
      },
      // Hide Summary fields and API keys in RJSF; custom section renders these
      prompt_type: { 'ui:widget': 'hidden' },
      summary_models: { 'ui:widget': 'hidden' },
      summary_max_tokens: { 'ui:widget': 'hidden' },
      custom_prompt: { 'ui:widget': 'hidden' },
      anthropic_api_key: { 'ui:widget': 'hidden' },
      openai_api_key: { 'ui:widget': 'hidden' },
      openrouter_api_key: { 'ui:widget': 'hidden' },
      gemini_api_key: { 'ui:widget': 'hidden' },
      huggingface_api_key: { 'ui:widget': 'hidden' },
      input_type: { 'ui:widget': 'hidden' },
    }),
    [enableStudy, enableSummary, promptType, requiredProviders, enableImages, inputMode]
  );

  const handleInputModeChange = (nextMode: InputMode) => {
    setInputMode(nextMode);
    setFormData((current) => {
      const updated: any = { ...current, input_type: nextMode };
      if (nextMode === 'region_names') {
        const arr = Array.isArray(updated.atlas_names) ? (updated.atlas_names as string[]) : [];
        updated.atlas_names = arr.length > 0 ? [arr[0]] : ['harvard-oxford'];
      }
      return updated;
    });
  };

  const handleCoordEntryModeChange = (nextMode: CoordEntryMode) => {
    setCoordEntryMode(nextMode);
    if (nextMode === 'file' && (!coordsFile || coordsFile.trim().length === 0)) {
      setCoordsFile('examples/toy_coordinates.csv');
    }
  };

  const handleRegionNamesInput = (value: string) => {
    setRegionNamesText(value);
  };

  const applyYamlObject = (data: Record<string, any>) => {
    try {
      // Determine input type and fill inputs
      const inputType = typeof data.input_type === 'string' ? data.input_type : (Array.isArray(data.region_names) ? 'region_names' : 'coords');
      if (inputType === 'region_names') {
        setInputMode('region_names');
        const names = Array.isArray(data.region_names) ? data.region_names.filter((s: any) => typeof s === 'string' && s.trim()).join('\n') : '';
        setRegionNamesText(names || '');
      } else {
        setInputMode('coords');
        if (Array.isArray(data.coordinates) && data.coordinates.length > 0) {
          const text = data.coordinates
            .filter((t: any) => Array.isArray(t) && t.length === 3)
            .map((t: any) => t.map((n: any) => String(n)).join(', '))
            .join('\n');
          setCoordEntryMode('paste');
          setCoordinateText(text);
          setCoordsFile('');
        } else if (typeof data.coords_file === 'string' && data.coords_file.trim()) {
          setCoordEntryMode('file');
          setCoordsFile(data.coords_file.trim());
        } else {
          setCoordEntryMode('paste');
        }
      }

      // Determine toggles from outputs
      const outputs = Array.isArray(data.outputs) ? (data.outputs as string[]).map((s) => String(s)) : [];
      const hasImages = outputs.includes('images');
      const hasSummaries = outputs.includes('summaries');
      const hasStudies = outputs.includes('raw_studies');
      setEnableSummary(!!hasSummaries);
      setEnableStudy(!!(hasStudies || hasSummaries));

      // Build next formData
      setFormData((curr) => {
        const next: any = { ...curr };
        next.input_type = inputType;
        // Atlases & sources
        if (Array.isArray(data.atlas_names)) next.atlas_names = data.atlas_names;
        if (Array.isArray(data.sources)) next.sources = data.sources;
        // Radii
        if (data.region_search_radius !== undefined) next.region_search_radius = String(data.region_search_radius ?? '');
        if (data.study_search_radius !== undefined) next.study_search_radius = String(data.study_search_radius ?? '');
        // Save & export
        if (typeof data.working_directory === 'string') next.working_directory = data.working_directory;
        next.output_format = typeof data.output_format === 'string' ? data.output_format : '';
        next.output_name = typeof data.output_name === 'string' ? data.output_name : '';
        // Summaries
        if (typeof data.prompt_type === 'string') next.prompt_type = data.prompt_type;
        if (Array.isArray(data.summary_models)) next.summary_models = data.summary_models;
        if (data.summary_max_tokens !== undefined && data.summary_max_tokens !== null) next.summary_max_tokens = String(data.summary_max_tokens);
        if (typeof data.custom_prompt === 'string') next.custom_prompt = data.custom_prompt;
        // API keys if present
        ['anthropic_api_key','openai_api_key','openrouter_api_key','gemini_api_key','huggingface_api_key','email_for_abstracts'].forEach((k) => {
          if (typeof data[k] === 'string') next[k] = data[k];
        });
        // Images
        next.outputs = hasImages ? ['images'] : withoutImages(curr.outputs);
        next.image_backend = typeof data.image_backend === 'string' ? data.image_backend : (hasImages ? (curr.image_backend || 'ai') : null);
        next.image_model = typeof data.image_model === 'string' ? data.image_model : (hasImages ? (curr.image_model || 'stabilityai/stable-diffusion-2') : null);
        next.image_prompt_type = typeof data.image_prompt_type === 'string' ? data.image_prompt_type : (hasImages ? (curr.image_prompt_type || 'anatomical') : null);
        next.image_custom_prompt = typeof data.image_custom_prompt === 'string' ? data.image_custom_prompt : null;
        return next;
      });

      setImportStatus('success');
      setImportMessage('Config loaded from YAML.');
      setTimeout(() => setImportStatus('idle'), 2500);
    } catch (e) {
      console.error('Failed to apply YAML config', e);
      setImportStatus('error');
      setImportMessage('Failed to load YAML. Please check the file format.');
      setTimeout(() => setImportStatus('idle'), 3500);
    }
  };

  const handleYamlFileInput = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const text = String(reader.result || '');
        const obj = YAML.load(text) as Record<string, any>;
        if (obj && typeof obj === 'object') {
          applyYamlObject(obj);
        } else {
          throw new Error('Empty or invalid YAML');
        }
      } catch (err) {
        console.error('YAML parse error', err);
        setImportStatus('error');
        setImportMessage('Unable to parse YAML file.');
        setTimeout(() => setImportStatus('idle'), 3500);
      }
      // clear input
      event.target.value = '';
    };
    reader.onerror = () => {
      setImportStatus('error');
      setImportMessage('Unable to read file.');
      setTimeout(() => setImportStatus('idle'), 3500);
    };
    reader.readAsText(file);
  };

  const applyTemplate = (id: string) => {
    // Save snapshot for Reset functionality (restore state before applying template)
    setTemplateUndo({
      inputMode,
      coordEntryMode,
      coordinateText,
      coordsFile,
      regionNamesText,
      enableStudy,
      enableSummary,
      formData,
    });

    if (id === 'single-lookup') {
      // Single coordinate, atlas lookup only
      setInputMode('coords');
      setCoordEntryMode('paste');
      setCoordinateText('0, 24, 26'); // Anterior cingulate demo
      setEnableSummary(false);
      setEnableStudy(false);
      setFormData((curr) => ({
        ...curr,
        input_type: 'coords',
        atlas_names: ['harvard-oxford', 'juelich'],
        sources: [],
        working_directory: '',
        output_format: '',
        output_name: '',
        region_search_radius: '0.4',
        study_search_radius: '6',
        prompt_type: 'summary',
        summary_models: [],
        summary_max_tokens: '',
        custom_prompt: null,
        outputs: withoutImages(curr.outputs),
        image_backend: null,
        image_model: null,
        image_prompt_type: null,
        image_custom_prompt: null
      }));
    } else if (id === 'multi-with-summaries') {
      // Multiple coordinates with summaries
      setInputMode('coords');
      setCoordEntryMode('paste');
      setCoordinateText('30, -22, 50\n-28, -20, 48');
      setEnableStudy(true);
      setEnableSummary(true);
      setFormData((curr) => ({
        ...curr,
        input_type: 'coords',
        atlas_names: ['harvard-oxford'],
        sources: ['neurosynth', 'neuroquery'],
        working_directory: '',
        output_format: '',
        output_name: '',
        region_search_radius: '0.4',
        study_search_radius: '6',
        prompt_type: 'summary',
        summary_models: ['gemini-2.0-flash'],
        summary_max_tokens: '800',
        custom_prompt: null,
        outputs: withoutImages(curr.outputs),
        image_backend: null,
        image_model: null,
        image_prompt_type: null,
        image_custom_prompt: null
      }));
    } else if (id === 'regions-to-coords') {
      // Region names to coordinates (single atlas enforced)
      setInputMode('region_names');
      setRegionNamesText('Anterior cingulate cortex\nSuperior frontal gyrus');
      setEnableStudy(false);
      setEnableSummary(false);
      setFormData((curr) => ({
        ...curr,
        input_type: 'region_names',
        atlas_names: ['harvard-oxford'],
        sources: [],
        working_directory: '',
        output_format: '',
        output_name: '',
        region_search_radius: '0.4',
        study_search_radius: '6',
        prompt_type: 'summary',
        summary_models: [],
        summary_max_tokens: '',
        custom_prompt: null,
        outputs: withoutImages(curr.outputs),
        image_backend: null,
        image_model: null,
        image_prompt_type: null,
        image_custom_prompt: null
      }));
    } else if (id === 'coords-to-insights') {
      // Full insights: coords + studies + summaries + images
      setInputMode('coords');
      setCoordEntryMode('paste');
      setCoordinateText('40, -50, 30\n-42, -46, 28');
      setEnableStudy(true);
      setEnableSummary(true);
      setFormData((curr) => ({
        ...curr,
        input_type: 'coords',
        atlas_names: ['harvard-oxford', 'schaefer'],
        sources: ['neurosynth', 'neuroquery'],
        working_directory: '',
        output_format: 'json',
        output_name: 'insights.json',
        region_search_radius: '0.4',
        study_search_radius: '6',
        prompt_type: 'summary',
        summary_models: Array.isArray(curr.summary_models) && (curr.summary_models as string[]).length
          ? curr.summary_models as string[]
          : ['gemini-2.0-flash', 'claude-3-haiku'],
        summary_max_tokens: '1000',
        custom_prompt: null,
        outputs: ['images'], // enable images
        image_backend: typeof curr.image_backend === 'string' && curr.image_backend ? (curr.image_backend as string) : 'ai',
        image_model: typeof curr.image_model === 'string' && curr.image_model ? (curr.image_model as string) : 'stabilityai/stable-diffusion-2',
        image_prompt_type: typeof curr.image_prompt_type === 'string' && curr.image_prompt_type ? (curr.image_prompt_type as string) : 'anatomical',
        image_custom_prompt: null
      }));
    }
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
          updated.study_search_radius = null;
        }
        return updated;
      });
      return next;
    });
  };

  const toggleSummary = () => {
    setEnableSummary((prev) => {
      const next = !prev;
      // When enabling summaries, also enable studies automatically
      if (next) {
        setEnableStudy(true);
      }
      setFormData((current) => {
        const updated = { ...current };
        if (!next) {
          updated.prompt_type = null;
          updated.summary_models = null;
          updated.summary_max_tokens = null;
          updated.custom_prompt = null;
        } else {
          if (typeof updated.prompt_type !== 'string' || !updated.prompt_type) {
            updated.prompt_type = 'summary';
          }
          if (updated.prompt_type !== 'custom') {
            updated.custom_prompt = null;
          }
          if (!Array.isArray(updated.summary_models)) {
            updated.summary_models = [];
          }
        }
        return updated;
      });
      return next;
    });
  };

  const toggleImages = () => {
    setFormData((current) => {
      const updated: any = { ...current };
      const outputs = Array.isArray(updated.outputs) ? (updated.outputs as string[]) : [];
      const next = new Set(outputs);
      if (next.has('images')) {
        next.delete('images');
        updated.image_backend = null;
        updated.image_model = null;
        updated.image_prompt_type = null;
        updated.image_custom_prompt = null;
      } else {
        next.add('images');
        if (typeof updated.image_backend !== 'string' || !updated.image_backend) {
          updated.image_backend = 'ai';
        }
        if (typeof updated.image_model !== 'string' || !updated.image_model) {
          updated.image_model = 'stabilityai/stable-diffusion-2';
        }
        if (typeof updated.image_prompt_type !== 'string' || !updated.image_prompt_type) {
          updated.image_prompt_type = 'anatomical';
        }
        if (updated.image_prompt_type !== 'custom') {
          updated.image_custom_prompt = null;
        }
      }
      updated.outputs = Array.from(next);
      return updated;
    });
  };

  const configData = useMemo(() => {
    const payload: Record<string, unknown> = {
      ...formData
    };

    if (inputMode === 'coords') {
      payload.input_type = 'coords';
      payload.coordinates = coordEntryMode === 'paste' ? (coords.length ? coords : null) : null;
      // Require non-empty coords_file when in file mode
      payload.coords_file = coordEntryMode === 'file' ? (coordsFile && coordsFile.trim().length > 0 ? coordsFile : null) : null;
      payload.region_names = null;
    } else {
      payload.input_type = 'region_names';
      payload.coordinates = null;
      payload.coords_file = null;
      payload.region_names = regionNameList.length ? regionNameList : null;
    }

    if (!enableStudy) {
      payload.study_search_radius = null;
    }

    // Normalize outputs to match CLI subcommands
    const imagesSelected = Array.isArray(formData.outputs)
      ? (formData.outputs as string[]).includes('images')
      : false;
    const outputsNormalized: string[] = [];
    if (inputMode === 'coords') {
      outputsNormalized.push('region_labels');
    } else {
      outputsNormalized.push('mni_coordinates');
    }
    if (enableSummary) {
      outputsNormalized.push('summaries');
    }
    if (imagesSelected) {
      outputsNormalized.push('images');
    }
    // Include raw_studies whenever summaries or images or study toggle is active
    if (enableStudy || enableSummary || imagesSelected) {
      outputsNormalized.push('raw_studies');
    }
    payload.outputs = outputsNormalized;

    if (!enableSummary) {
      payload.prompt_type = null;
      payload.summary_models = null;
      payload.summary_max_tokens = null;
      payload.custom_prompt = null;
    } else {
      // Ensure summary_models is an array
      if (!Array.isArray(payload.summary_models)) {
        payload.summary_models = payload.summary_models ? [payload.summary_models] : [];
      }
      // Filter out empty strings and null values
      payload.summary_models = (payload.summary_models as string[]).filter((model: string) => 
        typeof model === 'string' && model.trim().length > 0
      );
      // Set to null if no valid models
      if ((payload.summary_models as string[]).length === 0) {
        payload.summary_models = null;
      }
      if (payload.prompt_type !== 'custom') {
        payload.custom_prompt = null;
      }
    }

    // If images are not requested, drop image configuration keys
    const outputsArr = outputsNormalized;
    const wantsImages = outputsArr.includes('images');
    if (!wantsImages) {
      payload.image_backend = null;
      payload.image_model = null;
      payload.image_prompt_type = null;
      payload.image_custom_prompt = null;
    } else {
      // When using nilearn-only backend, drop AI-specific fields
      const backend = typeof payload.image_backend === 'string' ? (payload.image_backend as string) : 'ai';
      if (backend === 'nilearn') {
        payload.image_model = null;
        payload.image_prompt_type = null;
        payload.image_custom_prompt = null;
      } else {
        // Ensure defaults and consistency
        const ipt = typeof payload.image_prompt_type === 'string' && (payload.image_prompt_type as string)
          ? (payload.image_prompt_type as string)
          : 'anatomical';
        payload.image_prompt_type = ipt;
        if (ipt !== 'custom') {
          payload.image_custom_prompt = null;
        }
        // normalize image_model to string or null
        if (typeof payload.image_model !== 'string') {
          payload.image_model = null;
        }
      }
    }

    const summaryTokenValue = payload.summary_max_tokens;
    if (typeof summaryTokenValue === 'string') {
      const trimmed = summaryTokenValue.trim();
      if (!trimmed) {
        payload.summary_max_tokens = null;
      } else {
        const parsed = Number(trimmed);
        payload.summary_max_tokens = Number.isFinite(parsed) ? Math.floor(parsed) : null;
      }
    }

    const radiusValue = payload.region_search_radius;
    if (typeof radiusValue === 'string') {
      const trimmed = radiusValue.trim();
      if (!trimmed) {
        payload.region_search_radius = null;
      } else {
        const parsed = Number(trimmed);
        payload.region_search_radius = Number.isFinite(parsed) ? parsed : null;
      }
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
      // Reorder keys so outputs is listed directly after inputs
      const data = configData as Record<string, unknown>;
      const ordered: Record<string, unknown> = {};
      const saveExportKeys = ['working_directory', 'output_format', 'output_name'];
      const addIf = (key: string) => {
        if (Object.prototype.hasOwnProperty.call(data, key)) {
          ordered[key] = data[key];
        }
      };

      // Inputs section
      addIf('input_type');
      if (data['input_type'] === 'coords') {
        // Show coordinates if present, otherwise coords_file if present
        addIf('coordinates');
        addIf('coords_file');
      } else if (data['input_type'] === 'region_names') {
        addIf('region_names');
      }

      // Outputs immediately after inputs
      addIf('outputs');

      // Append the rest in their existing order
      Object.keys(data).forEach((key) => {
        if (!Object.prototype.hasOwnProperty.call(ordered, key)) {
          // Defer Save & Export section keys to the very end
          if (!saveExportKeys.includes(key)) {
            ordered[key] = data[key];
          }
        }
      });

      // Finally append Save & Export keys at the very bottom, in this order
      saveExportKeys.forEach((key) => {
        if (Object.prototype.hasOwnProperty.call(data, key)) {
          ordered[key] = data[key];
        }
      });

      return YAML.dump(ordered, { lineWidth: 120 });
    } catch (error) {
      console.error('Unable to render YAML preview', error);
      return '# Unable to render YAML preview';
    }
  }, [configData]);

  const cliCommand = 'coord2region --config coord2region-config.yaml';

  // Simple shell escaping suitable for zsh: wrap in single quotes and escape existing ones
  const shellEscape = (val: unknown): string => {
    const s = String(val ?? '');
    if (s === '') return "''";
    // If it's a safe token (alphanumerics, underscore, dash, slash, dot, colon), no quotes
    if (/^[A-Za-z0-9_:\/.\-]+$/.test(s)) return s;
    return `'${s.replace(/'/g, "'\\''")}'`;
  };

  // Generate direct CLI command(s) mirroring coord2region subcommands from current selections
  const directCliCommands = useMemo(() => {
    try {
      const outputs = Array.isArray((configData as any).outputs) ? ((configData as any).outputs as string[]) : [];
      const outputKey = new Set(outputs.map((o) => String(o).toLowerCase()));
      const inputType = String((configData as any).input_type || 'coords').toLowerCase();
      // Determine subcommand name and required capability flags
      type Caps = { include_api: boolean; include_sources: boolean; include_image: boolean };
      let command = '';
      let caps: Caps = { include_api: false, include_sources: false, include_image: false };
      const keyStr = Array.from(outputKey).sort().join(',');
      const match = (arr: string[]) => arr.length === outputKey.size && arr.every((x) => outputKey.has(x));
      if (inputType === 'coords') {
        if (match(['region_labels'])) {
          command = 'coords-to-atlas'; caps = { include_api: false, include_sources: false, include_image: false };
        } else if (match(['region_labels', 'raw_studies'])) {
          command = 'coords-to-study'; caps = { include_api: false, include_sources: true, include_image: false };
        } else if (match(['region_labels', 'raw_studies', 'summaries'])) {
          command = 'coords-to-summary'; caps = { include_api: true, include_sources: true, include_image: false };
        } else if (match(['region_labels', 'raw_studies', 'images'])) {
          command = 'coords-to-image'; caps = { include_api: false, include_sources: true, include_image: true };
        } else if (match(['region_labels', 'raw_studies', 'summaries', 'images'])) {
          command = 'coords-to-insights'; caps = { include_api: true, include_sources: true, include_image: true };
        } else {
          // Unsupported combination
          return [] as string[];
        }
      } else if (inputType === 'region_names') {
        if (match(['mni_coordinates'])) {
          command = 'region-to-coords'; caps = { include_api: false, include_sources: false, include_image: false };
        } else if (match(['mni_coordinates', 'raw_studies'])) {
          command = 'region-to-study'; caps = { include_api: false, include_sources: true, include_image: false };
        } else if (match(['mni_coordinates', 'raw_studies', 'summaries'])) {
          command = 'region-to-summary'; caps = { include_api: true, include_sources: true, include_image: false };
        } else if (match(['mni_coordinates', 'raw_studies', 'images'])) {
          command = 'region-to-image'; caps = { include_api: false, include_sources: true, include_image: true };
        } else if (match(['mni_coordinates', 'raw_studies', 'summaries', 'images'])) {
          command = 'region-to-insights'; caps = { include_api: true, include_sources: true, include_image: true };
        } else {
          return [] as string[];
        }
      } else {
        return [] as string[];
      }

      const tokens: string[] = ['coord2region', command];

      // Inputs
      if (inputType === 'coords') {
        if (coordEntryMode === 'file' && coordsFile && coordsFile.trim()) {
          tokens.push('--coords-file', shellEscape(coordsFile.trim()));
        } else {
          for (const triplet of coords) {
            for (const num of triplet) tokens.push(String(num));
          }
        }
      } else {
        // region_names
        for (const name of regionNameList) tokens.push(shellEscape(name));
      }

      const cfg: any = configData as any;
      // Common flags: working dir
      if (cfg.working_directory) {
        tokens.push('--working-directory', shellEscape(cfg.working_directory));
      }

      // API keys when summaries included
      if (caps.include_api) {
        const apiMap: Record<string, string> = {
          gemini_api_key: '--gemini-api-key',
          openrouter_api_key: '--openrouter-api-key',
          openai_api_key: '--openai-api-key',
          anthropic_api_key: '--anthropic-api-key',
          huggingface_api_key: '--huggingface-api-key',
        };
        for (const [key, flag] of Object.entries(apiMap)) {
          if (cfg[key]) tokens.push(flag, shellEscape(cfg[key]));
        }
      } else if (caps.include_image) {
        // Images may still rely on Hugging Face key
        if (cfg.huggingface_api_key) {
          tokens.push('--huggingface-api-key', shellEscape(cfg.huggingface_api_key));
        }
      }

      // Study sources and email
      if (caps.include_sources) {
        const sources = Array.isArray(cfg.sources) ? (cfg.sources as string[]) : [];
        if (sources.length) tokens.push('--sources', shellEscape(sources.join(',')));
        if (cfg.email_for_abstracts) tokens.push('--email-for-abstracts', shellEscape(cfg.email_for_abstracts));
      }

      // Atlas names
      const atlasNames = Array.isArray(cfg.atlas_names) ? (cfg.atlas_names as string[]) : [];
      for (const name of atlasNames) tokens.push('--atlas', shellEscape(name));
      const atlasConfigs = (cfg.atlas_configs || {}) as Record<string, Record<string, string>>;
      for (const [name, conf] of Object.entries(atlasConfigs)) {
        if (typeof conf !== 'object' || !conf) continue;
        if (conf.atlas_url && conf.atlas_url !== name) tokens.push('--atlas-url', shellEscape(`${name}=${conf.atlas_url}`));
        if (conf.atlas_file && conf.atlas_file !== name) tokens.push('--atlas-file', shellEscape(`${name}=${conf.atlas_file}`));
      }

      // Output/export flags
      if (cfg.output_format) tokens.push('--output-format', shellEscape(cfg.output_format));
      if (cfg.output_name) tokens.push('--output-name', shellEscape(cfg.output_name));

      // Image options
      if (caps.include_image) {
        if (cfg.image_backend) tokens.push('--image-backend', shellEscape(cfg.image_backend));
        if (cfg.image_model) tokens.push('--image-model', shellEscape(cfg.image_model));
        if (cfg.image_prompt_type) tokens.push('--image-prompt-type', shellEscape(cfg.image_prompt_type));
        if (cfg.image_custom_prompt) tokens.push('--image-custom-prompt', shellEscape(cfg.image_custom_prompt));
      }

      return [tokens.join(' ')];
    } catch {
      return [] as string[];
    }
  }, [configData, coords, coordEntryMode, coordsFile, regionNameList]);

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

  const githubUrl = 'https://github.com/BabaSanfour/Coord2Region';
  const docsUrl = 'https://coord2region.readthedocs.io/en/latest/';

  return (
    <>
      {props.showHeaderNav !== false && (
      <header className="site-header" role="banner">
        <h1 className="site-title">
          Coord2Region:  Coordinates <em className="into">into</em> Insights
        </h1>
        <p className="site-subtitle">
          Transform brain coordinates into region names, related studies, AI summaries, and AI‑generated images — with optional region‑based workflows too.
        </p>
        <nav className="site-nav" aria-label="Primary">
          <div className="nav-buttons">
            <button type="button" className={clsx('nav-btn', viewMode === 'about' && 'nav-btn--active')} onClick={() => setViewMode('about')}>About</button>
            <button type="button" className={clsx('nav-btn', viewMode === 'builder' && 'nav-btn--active')} onClick={() => setViewMode('builder')}>Config Builder</button>
            <button type="button" className={clsx('nav-btn', viewMode === 'cloud' && 'nav-btn--active')} onClick={() => setViewMode('cloud')}>Cloud Runner</button>
            <a className="nav-link" href={docsUrl} target="_blank" rel="noreferrer">Documentation</a>
            <a className="nav-link" href={githubUrl} target="_blank" rel="noreferrer">View on GitHub</a>
          </div>
        </nav>
      </header>
      )}

      {props.showHeaderNav !== false && viewMode === 'about' && (
        <section className="about-section card" role="region" aria-label="About Coord2Region">
          <h4>About</h4>
          <p>
            Coord2Region turns MNI coordinates and atlas region names into actionable insights. It maps coordinates to atlas labels,
            finds related studies from open neuro datasets, generates concise AI‑powered summaries, and optionally produces
            reproducible images. This website provides a guided Config Builder to export a YAML you can run locally, along with links
            to docs and the code.
          </p>
          <p>
            Typical use cases include: annotating peaks in fMRI results, localizing iEEG/MEG sources, cross‑referencing coordinates with
            literature, and packaging results for sharing and reproducibility.
          </p>
        </section>
      )}

      {props.showHeaderNav !== false && viewMode === 'cloud' && (
        <section className="cloud-section card" role="region" aria-label="Cloud Runner">
          <h4>Cloud Runner</h4>
          <p>(Phase 2 under construction)</p>
        </section>
      )}

      {(props.showHeaderNav === false || viewMode === 'builder') && (
        <section className="config-builder">
          {/* Row 1: Input (left) and Output (right) */}
          {/* Input panel */}
          <div className="card card--panel">
              <div className="card-header">
                <h4>Input</h4>
                <p className="helper">Choose how you want to provide data for mapping.</p>
              </div>
              {/* Atlas selection moved to Coord2RegionConfig section */}
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
                    <div className="form-field">
                      <label htmlFor="coord-textarea" className="field-label tooltip" data-tooltip={tooltipFromSchema('coordinates')}>
                        {schema.properties?.coordinates && typeof schema.properties.coordinates === 'object'
                          ? (schema.properties.coordinates as SchemaProperty).title || 'Coordinates'
                          : 'Coordinates'}
                      </label>
                      <textarea
                        id="coord-textarea"
                        className="coord-textarea"
                        value={coordinateText}
                        onChange={(event) => setCoordinateText(event.target.value)}
                        placeholder="30, -22, 50"
                        rows={5}
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
                    <div className="form-field">
                      <label htmlFor="coord-file" className="field-label tooltip" data-tooltip={tooltipFromSchema('coords_file')}>
                        Coordinate file path
                      </label>
                      <input
                        id="coord-file"
                        type="text"
                        className="coord-input"
                        value={coordsFile}
                        onChange={(event) => setCoordsFile(event.target.value)}
                        placeholder="/path/to/coordinates.tsv"
                      />
                      {coordFileError ? (
                        <ul className="form-errors"><li>Path is required when using a coordinate file.</li></ul>
                      ) : (
                        <p className="helper">Provide a local path to a CSV/TSV/XLSX file.</p>
                      )}
                    </div>
                  )}
                </>
              ) : (
                <div className="form-field">
                  <label htmlFor="region-names-textarea" className="field-label tooltip" data-tooltip={tooltipFromSchema('region_names')}>
                    Region names
                  </label>
                  <textarea
                    id="region-names-textarea"
                    className="coord-textarea"
                    value={regionNamesText}
                    onChange={(event) => handleRegionNamesInput(event.target.value)}
                    placeholder={"Amygdala\nHippocampus"}
                    rows={5}
                  />
                  <p className="helper">
                    Enter one region per line. Parsed {regionNameList.length} region name{regionNameList.length === 1 ? '' : 's'}.
                  </p>
                </div>
              )}
            </div>

          {/* Output panel */}
          <div className="card card--panel">
              <div className="card-header">
                <h4>Output</h4>
                <p className="helper">Baseline is added automatically based on the selected input.</p>
              </div>
              <div className="output-pills" role="tablist" aria-label="Output options">
                <span className="pill pill--locked pill--active" aria-disabled="true" aria-selected="true">
                  {inputMode === 'coords' ? 'Region names' : 'Coordinates'}
                </span>
                <button
                  type="button"
                  className={clsx('pill', outputDetail === 'studies' && 'pill--active')}
                  aria-pressed={outputDetail === 'studies'}
                  aria-selected={outputDetail === 'studies'}
                  onClick={() => setOutputDetail('studies')}
                >
                  Studies
                </button>
                <button
                  type="button"
                  className={clsx('pill', outputDetail === 'summaries' && 'pill--active')}
                  aria-pressed={outputDetail === 'summaries'}
                  aria-selected={outputDetail === 'summaries'}
                  onClick={() => setOutputDetail('summaries')}
                >
                  Summaries
                </button>
                <button
                  type="button"
                  className={clsx('pill', outputDetail === 'images' && 'pill--active')}
                  aria-pressed={outputDetail === 'images'}
                  aria-selected={outputDetail === 'images'}
                  onClick={() => setOutputDetail('images')}
                >
                  Images
                </button>
              </div>

              {(() => {
                let title = 'Summaries';
                let desc = 'Generate concise AI‑powered summaries using your configured models.';
                let isOn = enableSummary;
                let toggle = toggleSummary;
                if (outputDetail === 'studies') {
                  title = 'Studies';
                  desc = 'Include related papers for each coordinate/region from selected sources.';
                  isOn = enableStudy;
                  toggle = toggleStudy;
                } else if (outputDetail === 'images') {
                  title = 'Images';
                  desc = 'Create images using AI and/or nilearn backends.';
                  isOn = enableImages;
                  toggle = toggleImages;
                }
                return (
                  <div className="card card--inline">
                    <div>
                      <h5>{title}</h5>
                      <p>{desc}</p>
                    </div>
                    <button
                      type="button"
                      className={clsx('switch', isOn && 'switch--on')}
                      aria-pressed={isOn}
                      onClick={toggle}
                    >
                      <span className="switch__knob" />
                      <span className="switch__label">{isOn ? 'Enabled' : 'Disabled'}</span>
                    </button>
                  </div>
                );
              })()}
            </div>

          {/* Row 2: Left config card and right preview */}
          <div className="card">
            {/* Quick section navigation */}
            <nav className="section-nav" id="config-nav" aria-label="Configuration sections">
              <a href="#atlas-section" className={clsx('section-nav__link', activeSection === 'atlas-section' && 'section-nav__link--active')}>🗺️ <span>Atlas</span></a>
              {enableStudy && (
                <a href="#studies-section" className={clsx('section-nav__link', activeSection === 'studies-section' && 'section-nav__link--active')}>📚 <span>Studies</span></a>
              )}
              {enableSummary && (
                <a href="#summaries-section" className={clsx('section-nav__link', activeSection === 'summaries-section' && 'section-nav__link--active')}>✨ <span>Summaries</span></a>
              )}
              {enableImages && (
                <a href="#images-section" className={clsx('section-nav__link', activeSection === 'images-section' && 'section-nav__link--active')}>🖼️ <span>Images</span></a>
              )}
              <a href="#outputs-section" className={clsx('section-nav__link', activeSection === 'outputs-section' && 'section-nav__link--active')}>💾 <span>Outputs</span></a>
            </nav>
              <div className="atlas-section" id="atlas-section" role="group" aria-label="Atlas selection">
              <div className="atlas-section__header">
                <h5><span className="section-icon" aria-hidden>🗺️</span> Atlas selection</h5>
                <p className="helper">Choose the atlas libraries you want to query. You can also add custom names, URLs, or local paths.</p>
                {inputMode === 'region_names' && (
                  <p className="helper">When using Region names, select exactly one atlas.</p>
                )}
                {inputMode === 'region_names' && (() => {
                  const count = Array.isArray(formData.atlas_names) ? (formData.atlas_names as string[]).length : 0;
                  if (count === 0) {
                    return <p className="status status--error" role="alert">Select exactly one atlas for Region names input.</p>;
                  }
                  if (count > 1) {
                    return <p className="status status--error" role="alert">Multiple atlases selected. Region names differ across atlases—pick just one.</p>;
                  }
                  return null;
                })()}
              </div>
              <div className="atlas-section__controls">
                <div className="form-field form-field--inline">
                  <label
                    htmlFor="region-search-radius"
                    className="field-label tooltip"
                    data-tooltip={tooltipFromSchema('region_search_radius')}
                  >
                    Region search radius
                  </label>
                  <input
                    id="region-search-radius"
                    type="text"
                    value={(formData.region_search_radius as string) || ''}
                    onChange={(e) => setFormData(curr => ({ ...curr, region_search_radius: e.target.value }))}
                    placeholder="0.4"
                  />
                </div>
              </div>
              <AtlasSelection
                selected={Array.isArray(formData.atlas_names) ? (formData.atlas_names as string[]) : []}
                onChange={(next) => setFormData((curr) => ({ ...curr, atlas_names: next }))}
                enforceSingle={inputMode === 'region_names'}
              />
              <div className="section-footer"><a className="back-to-top" href="#config-nav">Back to top ↑</a></div>
            </div>

            {/* Studies Search section (visible only when Studies are enabled) */}
            {enableStudy && (
              <div className="studies-section" id="studies-section" role="group" aria-label="Studies search">
                <div className="studies-section__header">
                  <h5><span className="section-icon" aria-hidden>📚</span> Studies Search</h5>
                  <p className="helper">Select the literature sources to query and set search options. All fields are required when Studies are enabled.</p>
                </div>
                <div className="studies-sources">
                  <details className="studies-group" open>
                    <summary className="studies-group__summary">
                      <span className="studies-group__title">Sources</span>
                      <span className="studies-group__chips">
                        {(Array.isArray(formData.sources) ? (formData.sources as string[]) : []).slice(0,3).map((s) => (
                          <span key={s} className="chip">{s}</span>
                        ))}
                        {Array.isArray(formData.sources) && (formData.sources as string[]).length > 3 && (
                          <span className="chip chip--more">+{(formData.sources as string[]).length - 3}</span>
                        )}
                      </span>
                      <span className="studies-group__meta">{Array.isArray(formData.sources) ? (formData.sources as string[]).length : 0}/{datasetSourceOptions.length}</span>
                      <button
                        type="button"
                        className="studies-group__toggle"
                        onClick={(e) => {
                          e.preventDefault();
                          setFormData(curr => {
                            const current = Array.isArray(curr.sources) ? (curr.sources as string[]) : [];
                            const hasMissing = datasetSourceOptions.some(opt => !current.includes(opt));
                            const next = hasMissing ? [...datasetSourceOptions] : [];
                            return { ...curr, sources: next };
                          });
                        }}
                      >
                        {(() => {
                          const count = Array.isArray(formData.sources) ? (formData.sources as string[]).length : 0;
                          const all = count === datasetSourceOptions.length;
                          return all ? `Clear all (${datasetSourceOptions.length})` : `Select all (${datasetSourceOptions.length})`;
                        })()}
                      </button>
                    </summary>
                    <div className="studies-cards">
                      {datasetSourceOptions.map((opt) => {
                        const checked = Array.isArray(formData.sources) ? (formData.sources as string[]).includes(opt) : false;
                        return (
                          <label key={opt} className={clsx('study-card', checked && 'is-selected')}>
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() => {
                                setFormData(curr => {
                                  const current = new Set(Array.isArray(curr.sources) ? (curr.sources as string[]) : []);
                                  current.has(opt) ? current.delete(opt) : current.add(opt);
                                  return { ...curr, sources: Array.from(current) };
                                });
                              }}
                            />
                            <span className="study-card__name">{opt}</span>
                          </label>
                        );
                      })}
                    </div>
                    {Array.isArray(formData.sources) && (formData.sources as string[]).length === 0 && (
                      <p className="form-errors" role="alert">Select at least one source.</p>
                    )}
                  </details>
                </div>

                <div className="studies-options mini-grid">
                  <div className="form-field">
                    <label className="field-label tooltip" htmlFor="study-search-radius" data-tooltip={tooltipFromSchema('study_search_radius')}>Study search radius</label>
                    <input
                      id="study-search-radius"
                      type="text"
                      value={typeof formData.study_search_radius === 'string' ? (formData.study_search_radius as string) : (typeof formData.study_search_radius === 'number' ? String(formData.study_search_radius) : '')}
                      onChange={(e) => setFormData(curr => ({ ...curr, study_search_radius: e.target.value }))}
                      placeholder={String((schema.properties?.study_search_radius as any)?.default ?? '6')}
                    />
                  </div>
                  <div className="form-field">
                    <label className="field-label tooltip" htmlFor="email-for-abstracts" data-tooltip={tooltipFromSchema('email_for_abstracts')}>Email for abstracts</label>
                    <input
                      id="email-for-abstracts"
                      type="text"
                      value={typeof formData.email_for_abstracts === 'string' ? (formData.email_for_abstracts as string) : ''}
                      onChange={(e) => setFormData(curr => ({ ...curr, email_for_abstracts: e.target.value }))}
                      placeholder="name@example.com"
                    />
                  </div>
                </div>
                <div className="section-footer"><a className="back-to-top" href="#config-nav">Back to top ↑</a></div>
              </div>
            )}

            {/* Generate Summaries section */}
            {enableSummary && (
              <div className="summaries-section" id="summaries-section" role="group" aria-label="Generate summaries">
                <div className="summaries-section__header">
                  <h5><span className="section-icon" aria-hidden>✨</span> Generate Summaries</h5>
                  <p className="helper">Choose a prompt type, add one or more models, and set the max tokens. API keys will appear when required by selected models.</p>
                </div>
                <div className="mini-grid">
                  <div className="form-field">
                    <label className="field-label tooltip" htmlFor="prompt-type" data-tooltip={tooltipFromSchema('prompt_type') || undefined}>Prompt Type</label>
                    <div className="select-wrap">
                      <select
                        id="prompt-type"
                        className="select select--compact"
                        value={typeof formData.prompt_type === 'string' && formData.prompt_type ? (formData.prompt_type as string) : 'summary'}
                        onChange={(e) => setFormData(curr => ({ ...curr, prompt_type: e.target.value }))}
                      >
                        {promptTypeOptions.map((opt) => (
                          <option key={opt.value} value={opt.value}>{opt.label}</option>
                        ))}
                      </select>
                    </div>
                    <p className="helper">Select a template; choose “Custom prompt” to write yours.</p>
                  </div>

                  <div className="form-field">
                    <label className="field-label tooltip" htmlFor="summary-max-tokens" data-tooltip={tooltipFromSchema('summary_max_tokens') || undefined}>Max tokens</label>
                    <input
                      id="summary-max-tokens"
                      type="text"
                      value={typeof formData.summary_max_tokens === 'string' ? (formData.summary_max_tokens as string) : (typeof formData.summary_max_tokens === 'number' ? String(formData.summary_max_tokens) : '')}
                      onChange={(e) => setFormData(curr => ({ ...curr, summary_max_tokens: e.target.value }))}
                      placeholder="1000"
                    />
                    <p className="helper">Optional. Leave blank to use the provider default.</p>
                    <p className="helper helper--spacer" aria-hidden="true"></p>
                  </div>

                  <div className="form-field mini-span-2">
                    <label className="field-label" htmlFor="summary-models">Summary Models</label>
                    <div id="summary-models">
                      {/* Chips for selected models */}
                      {Array.isArray(formData.summary_models) && (formData.summary_models as string[]).length > 0 && (
                        <div className="selected-items" style={{ marginBottom: 6 }}>
                          {(formData.summary_models as string[]).map((m) => (
                            <span key={m} className="selected-item">
                              {m}
                              <button type="button" className="remove-item" onClick={() => setFormData(curr => ({ ...curr, summary_models: (curr.summary_models as string[]).filter(x => x !== m) }))} aria-label={`Remove ${m}`}>×</button>
                            </span>
                          ))}
                        </div>
                      )}
                      {/* Input with datalist suggestions */}
                      <input
                        type="text"
                        list="summary-model-options"
                        placeholder="Type model name and press Enter to add"
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            e.preventDefault();
                            const value = (e.target as HTMLInputElement).value.trim();
                            if (!value) return;
                            setFormData(curr => {
                              const arr = Array.isArray(curr.summary_models) ? (curr.summary_models as string[]) : [];
                              return { ...curr, summary_models: arr.includes(value) ? arr : [...arr, value] };
                            });
                            (e.target as HTMLInputElement).value = '';
                          }
                        }}
                      />
                      <datalist id="summary-model-options">
                        {summaryModelOptions.map((opt) => (
                          <option key={opt.value} value={opt.value}>{opt.label}</option>
                        ))}
                      </datalist>
                      <p className="helper">Press Enter to add a model. You can use any identifier supported by your providers.</p>
                      {(!Array.isArray(formData.summary_models) || (formData.summary_models as string[]).length === 0) && (
                        <p className="form-errors" role="alert">Select at least one model to generate summaries.</p>
                      )}
                    </div>
                  </div>

                  {typeof formData.prompt_type === 'string' && formData.prompt_type === 'custom' && (
                    <div className="form-field mini-span-2">
                      <label className="field-label" htmlFor="custom-prompt">Custom prompt template</label>
                      <textarea
                        id="custom-prompt"
                        rows={5}
                        value={typeof formData.custom_prompt === 'string' ? (formData.custom_prompt as string) : ''}
                        onChange={(e) => setFormData(curr => ({ ...curr, custom_prompt: e.target.value }))}
                        placeholder="You are an expert neuroscientist..."
                      />
                      <p className="helper">Use {`{coord}`} for the coordinate if needed.</p>
                    </div>
                  )}
                </div>

                {/* API keys based on required providers */}
                {Array.isArray(formData.summary_models) && (formData.summary_models as string[]).length > 0 && (
                  <div className="mini-grid">
                    {(() => {
                      const models = (formData.summary_models as string[]);
                      const providers = Array.from(new Set(models.map((m) => modelToProvider[m]).filter(Boolean)));
                      const missing: string[] = [];
                      if (providers.includes('anthropic') && !(typeof formData.anthropic_api_key === 'string' && formData.anthropic_api_key.trim())) missing.push('Anthropic');
                      if (providers.includes('openai') && !(typeof formData.openai_api_key === 'string' && formData.openai_api_key.trim())) missing.push('OpenAI');
                      if (providers.includes('openrouter') && !(typeof formData.openrouter_api_key === 'string' && formData.openrouter_api_key.trim())) missing.push('OpenRouter');
                      if (providers.includes('gemini') && !(typeof formData.gemini_api_key === 'string' && formData.gemini_api_key.trim())) missing.push('Google Gemini');
                      if (providers.includes('huggingface') && !(typeof formData.huggingface_api_key === 'string' && formData.huggingface_api_key.trim())) missing.push('Hugging Face');
                      return (
                        <>
                          {providers.includes('anthropic') && (
                            <div className="form-field">
                              <label className="field-label" htmlFor="anthropic-key">Anthropic API Key</label>
                              <input id="anthropic-key" type="text" value={typeof formData.anthropic_api_key === 'string' ? (formData.anthropic_api_key as string) : ''} onChange={(e) => setFormData(curr => ({ ...curr, anthropic_api_key: e.target.value }))} placeholder="Enter your Anthropic API key" />
                            </div>
                          )}
                          {providers.includes('openai') && (
                            <div className="form-field">
                              <label className="field-label" htmlFor="openai-key">OpenAI API Key</label>
                              <input id="openai-key" type="text" value={typeof formData.openai_api_key === 'string' ? (formData.openai_api_key as string) : ''} onChange={(e) => setFormData(curr => ({ ...curr, openai_api_key: e.target.value }))} placeholder="Enter your OpenAI API key" />
                            </div>
                          )}
                          {providers.includes('openrouter') && (
                            <div className="form-field">
                              <label className="field-label" htmlFor="openrouter-key">OpenRouter API Key</label>
                              <input id="openrouter-key" type="text" value={typeof formData.openrouter_api_key === 'string' ? (formData.openrouter_api_key as string) : ''} onChange={(e) => setFormData(curr => ({ ...curr, openrouter_api_key: e.target.value }))} placeholder="Enter your OpenRouter API key" />
                            </div>
                          )}
                          {providers.includes('gemini') && (
                            <div className="form-field">
                              <label className="field-label" htmlFor="gemini-key">Google Gemini API Key</label>
                              <input id="gemini-key" type="text" value={typeof formData.gemini_api_key === 'string' ? (formData.gemini_api_key as string) : ''} onChange={(e) => setFormData(curr => ({ ...curr, gemini_api_key: e.target.value }))} placeholder="Enter your Google Gemini API key" />
                            </div>
                          )}
                          {providers.includes('huggingface') && (
                            <div className="form-field">
                              <label className="field-label" htmlFor="hf-key">Hugging Face API Key</label>
                              <input id="hf-key" type="text" value={typeof formData.huggingface_api_key === 'string' ? (formData.huggingface_api_key as string) : ''} onChange={(e) => setFormData(curr => ({ ...curr, huggingface_api_key: e.target.value }))} placeholder="Enter your Hugging Face API key" />
                            </div>
                          )}
                          {missing.length > 0 && (
                            <p className="form-errors" role="alert">Missing API key{missing.length > 1 ? 's' : ''}: {missing.join(', ')}.</p>
                          )}
                        </>
                      );
                    })()}
                  </div>
                )}
                <div className="section-footer"><a className="back-to-top" href="#config-nav">Back to top ↑</a></div>
              </div>
            )}

            {/* Generate Images section */}
            {enableImages && (
              <div className="images-section summaries-section" id="images-section" role="group" aria-label="Generate images">
                <div className="summaries-section__header">
                  <h5><span className="section-icon" aria-hidden>🖼️</span> Generate Images</h5>
                  <p className="helper">Choose a backend, select a model (for AI), and pick a prompt template. Nilearn renders standard anatomical slices without prompts.</p>
                </div>
                <div className="mini-grid">
                  {/* Backend occupies full width row to place Model + Prompt type on same level below */}
                  <div className="form-field mini-span-2">
                    <label className="field-label" htmlFor="image-backend">Image backend</label>
                    <select
                      id="image-backend"
                      value={typeof formData.image_backend === 'string' && formData.image_backend ? (formData.image_backend as string) : 'ai'}
                      onChange={(e) => setFormData(curr => ({ ...curr, image_backend: e.target.value }))}
                    >
                      <option value="ai">AI</option>
                      <option value="nilearn">nilearn</option>
                      <option value="both">Both</option>
                    </select>
                  </div>

                  {showImageAiOptions && (
                    <div className="form-field">
                      <label className="field-label" htmlFor="image-model-input">Image Model</label>
                      {/* Simple input with datalist suggestions */}
                      <input
                        id="image-model-input"
                        type="text"
                        defaultValue={typeof formData.image_model === 'string' ? (formData.image_model as string) : ''}
                        onChange={(e) => setFormData(curr => ({ ...curr, image_model: e.target.value }))}
                        list="image-model-options"
                        placeholder="stabilityai/stable-diffusion-2"
                      />
                      <datalist id="image-model-options">
                        {imageModelOptions.map((opt) => (
                          <option key={opt.value} value={opt.value}>{opt.label}</option>
                        ))}
                      </datalist>
                      <p className="helper">Type a model identifier or choose a suggestion.</p>
                    </div>
                  )}

                  {showImageAiOptions && (
                    <div className="form-field">
                      <label className="field-label" htmlFor="image-prompt-type">Image prompt type</label>
                      <div className="select-wrap">
                        <select
                          id="image-prompt-type"
                          className="select select--compact"
                          value={typeof formData.image_prompt_type === 'string' && formData.image_prompt_type ? (formData.image_prompt_type as string) : 'anatomical'}
                          onChange={(e) => setFormData(curr => ({ ...curr, image_prompt_type: e.target.value }))}
                        >
                          {imagePromptTypeOptions.map((opt) => (
                            <option key={opt.value} value={opt.value}>{opt.label}</option>
                          ))}
                        </select>
                      </div>
                      <p className="helper">Select a template; choose “Custom prompt” to write yours.</p>
                    </div>
                  )}

                  {showImageAiOptions && typeof formData.image_prompt_type === 'string' && formData.image_prompt_type === 'custom' && (
                    <div className="form-field mini-span-2">
                      <label className="field-label" htmlFor="image-custom-prompt">Custom image prompt template</label>
                      <textarea
                        id="image-custom-prompt"
                        rows={4}
                        value={typeof formData.image_custom_prompt === 'string' ? (formData.image_custom_prompt as string) : ''}
                        onChange={(e) => setFormData(curr => ({ ...curr, image_custom_prompt: e.target.value }))}
                        placeholder="Create a detailed anatomical illustration of the brain region at MNI coordinate {coordinate}..."
                      />
                      <p className="helper">Templates can use {'{coordinate}'}, {'{first_paragraph}'}, and {'{atlas_context}'} placeholders.</p>
                    </div>
                  )}
                </div>
                <div className="section-footer"><a className="back-to-top" href="#config-nav">Back to top ↑</a></div>
              </div>
            )}
            {/* Outputs mini-section moved to the end (always last) */}
            <Form
              schema={builderSchema}
              formData={formData}
              onChange={handleFormChange}
              validator={validator as unknown as any}
              uiSchema={uiSchema as unknown as UiSchema<FormState>}
              widgets={widgets}
              fields={fields}
              templates={{ FieldTemplate, TitleFieldTemplate, DescriptionFieldTemplate } as any}
              formContext={{ promptType }}
              liveValidate={false}
              noHtml5Validate
            >
              {/* Outputs mini-section: Working directory (full row) + Output format & Output name (side-by-side) */}
              <div className="mini-section" id="outputs-section">
                <h5><span className="section-icon" aria-hidden>💾</span> Outputs</h5>
                <p className="helper">Configure where datasets/atlases are saved and how exported files are named.</p>
                <div className="mini-grid">
                  <div className="form-field mini-span-2">
                    <label className="field-label" htmlFor="mini-working-dir">Working directory</label>
                    <input
                      id="mini-working-dir"
                      type="text"
                      value={(formData.working_directory as string) || ''}
                      onChange={(e) => setFormData(curr => ({ ...curr, working_directory: e.target.value || null }))}
                      placeholder="/path/to/working-directory"
                    />
                    <p className="helper">Used for caches and downloads. Reuse this folder across runs to avoid recomputation and re‑downloads.</p>
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="mini-output-format">Output format</label>
                    <select
                      id="mini-output-format"
                      value={(formData.output_format as string) || ''}
                      onChange={(e) => setFormData(curr => ({ ...curr, output_format: e.target.value || null }))}
                    >
                      <option value="">No export</option>
                      {outputFormatOptions.map((option) => (
                        <option key={option} value={option}>{option.toUpperCase()}</option>
                      ))}
                    </select>
                    <p className="helper">Leave empty to skip file export.</p>
                  </div>
                  <div className="form-field">
                    <label className="field-label" htmlFor="mini-output-name">Output name</label>
                    <input
                      id="mini-output-name"
                      type="text"
                      value={(formData.output_name as string) || ''}
                      onChange={(e) => setFormData(curr => ({ ...curr, output_name: e.target.value || null }))}
                      placeholder="results.json"
                    />
                    <p className="helper">File name used for exports.</p>
                  </div>
                </div>
                <div className="section-footer"><a className="back-to-top" href="#config-nav">Back to top ↑</a></div>
              </div>
              <div className="form-footer">
                <small>Changes apply immediately to the YAML preview and CLI helpers.</small>
              </div>
            </Form>
          </div>

          <aside className="config-preview">
            <div className="card">
              <div className="preview-header">
                <h4>YAML preview</h4>
                <div className="config-actions">
                  <button
                    type="button"
                    onClick={() => copyToClipboard(yamlPreview, setYamlCopied)}
                    disabled={coordFileError}
                    aria-disabled={coordFileError || undefined}
                  >
                    Copy YAML
                  </button>
                  <button type="button" onClick={handleDownload} disabled={coordFileError} aria-disabled={coordFileError || undefined}>Download YAML</button>
                </div>
              </div>
              <p className="helper">Live YAML that updates as you edit. Save or copy to use with the CLI.</p>
              <pre className="yaml-output" aria-live="polite">
                <code>{yamlPreview}</code>
              </pre>
              {yamlCopied === 'copied' && <p className="status status--success">YAML copied to clipboard.</p>}
              {yamlCopied === 'error' && <p className="status status--error">Unable to copy YAML automatically.</p>}
              {coordFileError && (
                <p className="status status--error">Coordinate file path is required when using file input.</p>
              )}
            </div>
            <div className="card">
              <div className="preview-header">
                <h4>CLI command</h4>
                <div className="config-actions">
                  <button
                    type="button"
                    onClick={() => copyToClipboard(cliCommand, setCliCopied)}
                    disabled={coordFileError}
                    aria-disabled={coordFileError || undefined}
                  >
                    Copy command
                  </button>
                </div>
              </div>
              <p className="helper">Runs <code>coord2region</code> using a saved YAML file. Save or copy the YAML above, then run this command in your terminal.</p>
              <code className="cli-command">{cliCommand}</code>
              {cliCopied === 'copied' && <p className="status status--success">Command copied.</p>}
              {cliCopied === 'error' && <p className="status status--error">Unable to copy command.</p>}
            </div>
            <div className="card">
              <div className="preview-header">
                <h4>Direct CLI command</h4>
                <div className="config-actions">
                  <button
                    type="button"
                    onClick={() => directCliCommands[0] && copyToClipboard(directCliCommands[0], setDirectCliCopied)}
                    disabled={coordFileError || directCliCommands.length === 0}
                    aria-disabled={coordFileError || directCliCommands.length === 0 || undefined}
                  >
                    Copy direct command
                  </button>
                </div>
              </div>
              <p className="helper">Generates a direct subcommand from your selections—no YAML file required.</p>
              {directCliCommands.length === 0 ? (
                <p className="helper">Enable an output to generate a direct command for your current configuration.</p>
              ) : (
                <code className="direct-cli">{directCliCommands[0]}</code>
              )}
              {directCliCopied === 'copied' && <p className="status status--success">Command copied.</p>}
              {directCliCopied === 'error' && <p className="status status--error">Unable to copy command.</p>}
            </div>
            <div className="card">
              <div className="card-header">
                <h4>Templates & Import</h4>
                <p className="helper">Load a complete example configuration, or import a local YAML to continue where you left off.</p>
                <div className="template-toolbar" role="group" aria-label="Load example template or import YAML">
                  <label htmlFor="template-select" className="sr-only">Select a template</label>
                  <div className="select-wrap">
                    <select
                      id="template-select"
                      className="select select--compact"
                      value={selectedTemplate}
                      onChange={(e) => setSelectedTemplate(e.target.value)}
                    >
                      <option value="">Choose an example…</option>
                      <optgroup label="Examples">
                        <option value="single-lookup">📍 Single coordinate — atlas lookup</option>
                        <option value="multi-with-summaries">📚 Multiple coordinates + summaries</option>
                        <option value="regions-to-coords">🧠 Region names → coords</option>
                        <option value="coords-to-insights">🖼️ Coords → studies, summaries, images</option>
                      </optgroup>
                    </select>
                  </div>
                  <div className="template-actions">
                    <button
                      type="button"
                      className="template-btn template-btn--primary"
                      onClick={() => selectedTemplate && applyTemplate(selectedTemplate)}
                      disabled={!selectedTemplate}
                    >
                      <span aria-hidden>📦</span>
                      <span>Load template</span>
                    </button>
                    {(
                      <button
                        type="button"
                        className="template-btn template-btn--ghost"
                        onClick={() => {
                          if (templateUndo) {
                            // Restore snapshot
                            setInputMode(templateUndo.inputMode);
                            setCoordEntryMode(templateUndo.coordEntryMode);
                            setCoordinateText(templateUndo.coordinateText);
                            setCoordsFile(templateUndo.coordsFile);
                            setRegionNamesText(templateUndo.regionNamesText);
                            setEnableStudy(templateUndo.enableStudy);
                            setEnableSummary(templateUndo.enableSummary);
                            setFormData(templateUndo.formData);
                            setTemplateUndo(null);
                          }
                          setSelectedTemplate('');
                        }}
                        disabled={!templateUndo && !selectedTemplate}
                      >
                        Reset
                      </button>
                    )}
                  </div>
                  {/* Separate row for YAML import */}
                  <div className="template-yaml">
                    <input
                      id="yaml-file-input"
                      type="file"
                      accept=".yml,.yaml,text/yaml,text/x-yaml,application/x-yaml"
                      onChange={handleYamlFileInput}
                      style={{ display: 'none' }}
                    />
                    <button
                      type="button"
                      className="template-btn template-btn--secondary"
                      onClick={() => {
                        const el = document.getElementById('yaml-file-input') as HTMLInputElement | null;
                        el?.click();
                      }}
                    >
                      Load YAML…
                    </button>
                    {importStatus === 'success' && <span className="status status--success">{importMessage}</span>}
                    {importStatus === 'error' && <span className="status status--error">{importMessage}</span>}
                  </div>
                </div>
              </div>
            </div>
          </aside>

        </section>
      )}
    </>
  );
};

export default ConfigBuilder;
