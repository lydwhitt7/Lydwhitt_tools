import pandas as pd
import numpy as np

# Define oxide data in a dictionary, the properties of each oxide are defined here so they only need to be edited once.
oxide_data = {
    'SiO2': {'element_num': 1, 'ox_num': 2, 'atomic_mass': 28.086},
    'TiO2': {'element_num': 1, 'ox_num': 2, 'atomic_mass': 47.867},
    'Al2O3': {'element_num': 2, 'ox_num': 3, 'atomic_mass': 26.982},
    'FeOt': {'element_num': 1, 'ox_num': 1, 'atomic_mass': 55.845},
    'MgO': {'element_num': 1, 'ox_num': 1, 'atomic_mass': 24.305},
    'CaO': {'element_num': 1, 'ox_num': 1, 'atomic_mass': 40.078},
    'Na2O': {'element_num': 2, 'ox_num': 1, 'atomic_mass': 22.990},
    'K2O': {'element_num': 2, 'ox_num': 1, 'atomic_mass': 39.098},
    'Cr2O3': {'element_num': 2, 'ox_num': 3, 'atomic_mass': 52.00},
    'MnO': {'element_num': 1, 'ox_num': 1, 'atomic_mass': 54.94}
}

O_atomic_mass = 15.999
Fe_molar_mass = 55.845
FeO_molar_mass = 71.844
Fe2O3_molar_mass = 159.687

# Compute molecular weights
for oxide, data in oxide_data.items(): #for *key*, *value* in *dictionary*.items (when you use .items() (in a loop youre unpacking tuples in a dictionary into two variables. The first variable recieves the key youve assigned and the second recieves the value.)
    data['molecular_weight'] = (data['ox_num'] * O_atomic_mass) + (data['element_num'] * data['atomic_mass'])

phases = {
    'Ol': {'oxygen_target': 4, 'suffix': "_Ol", 'norm': "apfu", 'indices': ["Fo_num", "Fa_num", "Mg_num"]},
    'Cpx': {'oxygen_target': 6, 'suffix': "_Cpx", 'norm': "apfu", 'indices': ["Mg_num", "Wo", "En", "Fs", "Jd", "CaTs"]},
    'Plg': {'oxygen_target': 8, 'suffix': "_Plg", 'norm': "apfu", 'indices': ["An_num", "Ab_num", "Or_num"]},
    'Liq': {'oxygen_target': None, 'suffix': "_Liq", 'norm': "cation_fraction", 'indices': ["Mg_num"]}
}

#Func.1. is recalculating to FeOt regardless of input data. I have plans to edit this variable in future so you can select what you use but thats a future idea. 
def recalc_Fe(df):
    df = df.copy()
    if 'FeO' in df.columns and 'Fe2O3' in df.columns: #columns is identifying were looking for a column heading
        df.loc[df['FeO'].notna() & df['Fe2O3'].notna(), 'FeOt'] = df['FeO'] + df['Fe2O3'] * 0.8998 # this line is structured like this: df.loc[<condition>, 'FeOt'] so the condition is having a value in the FeO and Fe2O3 columns. if this condition is met then the column FeOT is targeted for modification, the modification given comes after the equals. 
    elif 'Fe2O3t' in df.columns:
        df.loc[df['Fe2O3t'].notna(), 'FeOt'] = df['Fe2O3t'] * 0.8998
    elif 'FeO' in df.columns:
        df.loc[df['FeO'].notna(), 'FeOt'] = df['FeO']
    return df.drop(columns=['Fe2O3t', 'Fe2O3', 'FeO'], errors='ignore')#this "errors" part of the code exists to ensure that if one of those columns doesnt exist to drop it doesnt become a problem. 

#Func.2. normallise data anhdrous- only applicable for liq data
def norm_anhy(df):
    present_oxides = [ox for ox in oxide_data if ox in df.columns]#this line is checking which oxides from our dictionary are present in the dataframe assigned to the function. Ox is the key assigned to identify each oxide in the dictionary. 
    df['Total'] = df[present_oxides].sum(axis=1)  # This line sums all the present oxides for each row
    df['norm_factor'] = 100 / df['Total']
    for ox in present_oxides:
        df[ox] = df[ox] * df['norm_factor']
    return df

#Func.3. normallise per oxygen (determined by phase)
def norm_ox(df, phase):
    #check "phase" input is one of the options we currently have
    if phase not in phases:
        raise ValueError(
            f"Unknown phase '{phase}'. "
            f"Valid options are: {', '.join(phases.keys())}"
        )
    mineral = phases[phase] #defines mineral to get phase specific settings fro the phases dictionary. 
    oxygen_target = mineral['oxygen_target']
    if oxygen_target is None:
        raise ValueError(f"APFU normalization is for minerals only. Phase '{phase}' has no fixed oxygen target.")

    #create list to store value
    oxygen_cols = []   

    for ox, props in oxide_data.items():
        if ox in df.columns:
            mols    = (df[ox] / props['molecular_weight']) * props['element_num']
            oxygens = (df[ox] / props['molecular_weight']) * props['ox_num']

            base = ox.split('O')[0]
            el_name = base.rstrip('0123456789')
            df[el_name + '_mol'] = mols

            oxygen_cols.append(oxygens)

    if not oxygen_cols:
        raise ValueError("No valid oxide columns found (expected names like 'SiO2', 'Al2O3', 'FeOt').")

    # Row-wise sum that preserves NaNs if any contributor is NaN
    df['O_sum'] = pd.concat(oxygen_cols, axis=1).sum(axis=1, skipna=True)

    # Optionally: blank rows with zero oxygen total (avoid div/0)
    df.loc[df['O_sum'] == 0, 'O_sum'] = np.nan

    norm_factor = df['O_sum'] / oxygen_target

    for ox in oxide_data:
        base = ox.split('O')[0]
        el_name = base.rstrip('0123456789')
        mol_col = el_name + '_mol'
        if mol_col in df.columns:
            df[el_name] = df[mol_col] / norm_factor

    return df


#Func.4. cation fraction calculation for liq data. 
def recalc_cat(df):
    elements = []

    for ox, data in oxide_data.items():

        base = ox.split('O')[0]
        el = base.rstrip('0123456789')

        mol_col = f"{el}_mol" 

        if ox in df.columns:
            
            df[mol_col] = (df[ox] / data['molecular_weight']) * data['element_num']
            elements.append(el)  # CHANGED

    
    mol_cols = [f"{el}_mol" for el in elements]

    if mol_cols:
        df['cat_total'] = df[mol_cols].sum(axis=1)
        df.loc[df['cat_total'] == 0, 'cat_total'] = np.nan 
       
        for el in elements:
            mol_col = f"{el}_mol"
            df[el] = df[mol_col] / df['cat_total']
    else:
        df['cat_total'] = np.nan 

    if 'Mg' in df.columns and 'Fe' in df.columns:
        df['Mg_num'] = (df['Mg'] / (df['Mg'] + df['Fe'])) * 100

    for ox, data in oxide_data.items():
        ox_mol_col = ox + '_true_mol'  # new column for true mols of oxide
        if ox in df.columns:
            df[ox_mol_col] = df[ox] / data['molecular_weight']

    return df

def recalc(df, phase, anhydrous=True, mol_values=True):

    #actions on all phases
    df = df.copy()
    if phase not in phases:
        raise ValueError(f"Unknown phase '{phase}'. Valid options are: {', '.join(phases.keys())}")
    
    mineral = phases[phase]
    df = df.rename(columns={col: col.replace(mineral['suffix'], '') 
                            for col in df.columns if mineral['suffix'] in col})

    #Recalculate the iron component to FeOt
    df = recalc_Fe(df)

    #action for if 'Liq' phase
    if phase == 'Liq':
        if anhydrous:
            df = norm_anhy(df)
        df = recalc_cat(df)

    #action for if mineral phase
    else:
        df = norm_ox(df, phase)

        if phase == 'Ol':
            df['Fo_num'] = 100 * df['Mg'] / (df['Mg'] + df['Fe'])
            df['Fa_num'] = 100 - df['Fo_num']
            df['Mg_num'] = df['Fo_num']
        elif phase == 'Cpx':
            total = (df['Mg'] + df['Fe'] + df.get('Ca', 0))
            total = total.replace(0, np.nan)
            df['En'] = df['Mg'] / total
            df['Fs'] = df['Fe'] / total
            df['Wo'] = df.get('Ca', 0) / total
            df['Mg_num'] = 100 * df['Mg'] / (df['Mg'] + df['Fe'])
        elif phase == 'Plg':
            alk_total = (df.get('Ca', 0) + df.get('Na', 0) + df.get('K', 0))
            alk_total = alk_total.replace(0, np.nan)
            df['An_num'] = 100 * df.get('Ca', 0) / alk_total
            df['Ab_num'] = 100 * df.get('Na', 0) / alk_total
            df['Or_num'] = 100 * df.get('K', 0)  / alk_total
    
    # Add suffix back to the original oxide columns
    original_oxide_cols = [col for col in df.columns if col in oxide_data.keys()]
    rename_back = {col: col + mineral['suffix'] for col in original_oxide_cols}
    df = df.rename(columns=rename_back)

    if mol_values:
        df = df.drop(columns=[col for col in df.columns if col.endswith('_mol')], errors='ignore')

    return df