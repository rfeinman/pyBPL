library_folder = 'library/';
fields = fieldnames(lib);

for i=1:numel(fields)
  name = fields{i}
  value = lib.(name);
  save_file(name, 'value');
  if isstruct(value)
    fields1 = fieldnames(value);
    for j=1:numel(fields1)
        name1 = fields1{j};
        value1 = value.(name1);
        name_total = strcat(name, '-', name1);
        save_file(name_total, value1);
    end
  elseif isa(value, 'SpatialModel')
      save_file('Spatial-last_model_id', value.last_model_id);
  else
    save_file(name, value);
  end
end

function save_file(fname, value)
    fname = strcat(library_folder, fname, '.mat');
    save(fname, 'value');
end