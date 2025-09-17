import { useCallback, useMemo, useState } from 'react';
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

type CoordMode = 'coordinates' | 'file';

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

const atlasSuggestions = [
  'harvard-oxford',
  'juelich',
  'aal',
  'schaefer',
  'brainnetome',
  'destrieux'
];

const builderKeys: string[] = [
  'input_type',
  'data_dir',
  'dataset_sources',
  'atlas_names',
  'use_atlases',
  'max_atlases',
  'region_names',
  'outputs',
  'output_format',
  'output_path',
  'studies',
  'study_sources',
  'study_radius',
  'study_limit',
  'summary_type',
  'summary_model',
  'summary_max_tokens',
  'summary_cache_size',
  'summary_prompt_template',
  'use_cached_dataset',
  'batch_size',
  'anthropic_api_key',
  'openai_api_key',
  'openrouter_api_key',
  'gemini_api_key',
  'huggingface_api_key',
  'email_for_abstracts'
];

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
  const { id, classNames, label, required, description, errors, help, children } = props;
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
  const selected = Array.isArray(value) ? (value as string[]) : [];
  const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const options = Array.from(event.target.selectedOptions).map((option) => option.value);
    onChange(options);
  };

  const handleCustomAtlas = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const form = event.currentTarget;
    const input = form.elements.namedItem('customAtlas') as HTMLInputElement | null;
    if (!input) {
      return;
    }
    const atlas = input.value.trim();
    if (atlas && !selected.includes(atlas)) {
      onChange([...selected, atlas]);
    }
    input.value = '';
  };

  const suggestions = Array.from(new Set([...atlasSuggestions, ...selected]));

  return (
    <div className="atlas-widget">
      <select
        id={id}
        className="atlas-widget__select"
        multiple
        disabled={disabled || readonly}
        value={selected}
        onChange={handleChange}
      >
        {suggestions.map((atlas) => (
          <option key={atlas} value={atlas}>
            {atlas}
          </option>
        ))}
      </select>
      <form className="atlas-widget__form" onSubmit={handleCustomAtlas}>
        <input
          type="text"
          name="customAtlas"
          placeholder="Add atlas by name"
          disabled={disabled || readonly}
        />
        <button type="submit" disabled={disabled || readonly}>
          Add
        </button>
      </form>
      {selected.length === 0 && <p className="atlas-widget__hint">Select one or more atlases to query.</p>}
    </div>
  );
};

const SummaryModelField = ({ formData, onChange, idSchema }: FieldProps) => {
  const value = typeof formData === 'string' ? formData : '';
  const inputId = idSchema?.$id ?? 'summary-model';
  return (
    <div className="form-field">
      <label className="field-label" htmlFor={inputId}>Summary Model</label>
      <input
        id={inputId}
        type="text"
        value={value}
        placeholder="e.g. gpt-4o"
        onChange={(event) => onChange(event.target.value || null)}
      />
    </div>
  );
};

const widgets = {
  atlasMultiSelect: AtlasMultiSelect
};

const fields = {
  summaryModelField: SummaryModelField
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
  const [mode, setMode] = useState<CoordMode>('coordinates');
  const [coordinateText, setCoordinateText] = useState('30, -22, 50');
  const [coordsFile, setCoordsFile] = useState('');
  const [enableStudy, setEnableStudy] = useState(false);
  const [enableSummary, setEnableSummary] = useState(true);
  const [yamlCopied, setYamlCopied] = useState<'idle' | 'copied' | 'error'>('idle');
  const [cliCopied, setCliCopied] = useState<'idle' | 'copied' | 'error'>('idle');

  const [formData, setFormData] = useState<FormState>(() => {
    const defaults = deriveDefaults(builderKeys);
    if (!defaults.outputs) {
      defaults.outputs = ['region_labels'];
    }
    if (!defaults.atlas_names) {
      defaults.atlas_names = ['harvard-oxford', 'juelich'];
    }
    defaults.input_type = 'coords';
    return defaults;
  });

  const { coords, errors: coordErrors } = useMemo(
    () => parseCoordinateText(coordinateText),
    [coordinateText]
  );

  const uiSchema: UiSchema = useMemo(
    () => ({
      'ui:order': builderKeys,
      atlas_names: {
        'ui:widget': 'atlasMultiSelect'
      },
      outputs: {
        'ui:widget': 'checkboxes'
      },
      studies: enableStudy ? {} : { 'ui:widget': 'hidden' },
      study_sources: enableStudy ? {} : { 'ui:widget': 'hidden' },
      study_radius: enableStudy ? {} : { 'ui:widget': 'hidden' },
      study_limit: enableStudy ? {} : { 'ui:widget': 'hidden' },
      summary_type: enableSummary ? {} : { 'ui:widget': 'hidden' },
      summary_model: enableSummary
        ? { 'ui:field': 'summaryModelField' }
        : { 'ui:widget': 'hidden' },
      summary_max_tokens: enableSummary ? {} : { 'ui:widget': 'hidden' },
      summary_cache_size: enableSummary ? {} : { 'ui:widget': 'hidden' },
      summary_prompt_template: enableSummary ? {} : { 'ui:widget': 'hidden' },
      input_type: { 'ui:widget': 'hidden' }
    }),
    [enableStudy, enableSummary]
  );

  const handleModeChange = (nextMode: CoordMode) => {
    setMode(nextMode);
    setFormData((current) => ({
      ...current,
      input_type: nextMode === 'coordinates' ? 'coords' : 'file'
    }));
  };

  const handleFormChange = useCallback((event: IChangeEvent<FormState>) => {
    setFormData(event.formData as FormState);
  }, []);

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
          updated.summary_type = null;
          updated.summary_model = null;
          updated.summary_max_tokens = null;
          updated.summary_cache_size = null;
          updated.summary_prompt_template = null;
        }
        return updated;
      });
      return next;
    });
  };

  const configData = useMemo(() => {
    const payload: Record<string, unknown> = {
      ...formData,
      coordinates: mode === 'coordinates' ? (coords.length ? coords : null) : null,
      coords_file: mode === 'file' ? (coordsFile || null) : null
    };

    if (!enableStudy) {
      payload.studies = null;
      payload.study_sources = null;
      payload.study_radius = null;
      payload.study_limit = null;
    }

    if (!enableSummary) {
      payload.summary_type = null;
      payload.summary_model = null;
      payload.summary_max_tokens = null;
      payload.summary_cache_size = null;
      payload.summary_prompt_template = null;
    }

    return (sanitizeValue(payload) as Record<string, unknown>) ?? {};
  }, [formData, mode, coords, coordsFile, enableStudy, enableSummary]);

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
        <h4>Choose how to provide coordinates</h4>
        <div className="mode-toggle" role="radiogroup" aria-label="Coordinate input mode">
          <button
            type="button"
            className={clsx('toggle', mode === 'coordinates' && 'toggle--active')}
            onClick={() => handleModeChange('coordinates')}
            aria-checked={mode === 'coordinates'}
            role="radio"
          >
            Paste coordinates
          </button>
          <button
            type="button"
            className={clsx('toggle', mode === 'file' && 'toggle--active')}
            onClick={() => handleModeChange('file')}
            aria-checked={mode === 'file'}
            role="radio"
          >
            Use coordinate file
          </button>
        </div>
        {mode === 'coordinates' ? (
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
