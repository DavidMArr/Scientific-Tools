# -*- coding: utf-8 -*-
"""
@author: David Magalhaes Sousa
@email: davidmagalhaessousa@gmail.com
"""

import numpy as np
import pandas as pd
import os
import time
from matplotlib import pyplot as plt
from operator import itemgetter
from bs4 import BeautifulSoup
from requests import get
from json import loads
# conda install -c conda-forge pymatgen
# or
# pip install pymatgen
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import gc # Required ot clean the patterns variable if the user does not want to accumulate (stack) patterns


#######Put your own API key in########
api_key = MPRester("ADD YOUR KEY HERE")
######################################


# Global variable to capture the results, in case the user forgets to attribute the function result to a variable:
patterns = [["Material id", "Formula", "Spacegroup", "Crystal system", "X-ray diffraction pattern", "Band gap/eV", "Final energy/eV", "Density (atomic)", "Density/(g/cm3)", "Metal's atomic weight/(g/mol)"]]


def XRD_PLOTTER(data_list : list, material_id_list : list = [], element_criteria = ["Fe", "S"], criteria_add = {'nsites': {'$lt': 1000000}, "theoretical": False}, data_labels : list = [], selected = 1, xlim = [10,90], stretch_references = False, stretch_thresh = 10, stack_patterns = False, axis_width = 70, axis_height = 20):
    """
    Description
    -------
    Searches materialsproject database for reference patterns containing a set of elements and plots them against the user data.
    
    Parameters
    ----------
    data_list : list
        List of data with the following example format: [[x1, y1], [x2, y2], ...].
        XRDAnalysis.py supplies a read_csv definition for CENIMAT (NOVA School of Science and Technology) XRD patterns. If a file is used, it will automatically assume read_csv is to be used on it.
    material_id_list : list
        List of material_id_list. Leave list empty if you prefer to generate a list of IDs based on the following arguments. It can also be generated with the following example, which yields the materials sorted based on formation_energy_per_atom from the lowest value (negative) to the highest:
            entries          = entries_chemsys(['In', 'Sn', 'S'])
            energy_per_atom  = [a["formation_energy_per_atom"] for a in entries if a["nelements"] > 1]
            material_ids     = [a["material_id"] for a in entries if a["nelements"] > 1]
            material_id_list = [x for _,x in sorted(zip(energy_per_atom, material_ids), reverse=False)]
        or/and using the argument elements.
    element_criteria : list
        List of elements to generate a list of "material_id" that includes X-ray diffraction patterns.
    criteria_add : dict
        Dictionary to be concatenated to the query criteria. Choose criteria_add = {}, for no aditional criteria.
    data_labels : list
        List of the data labels. Must match the size of the list data_list.
    selected : int
        Selected data to plot on tp of the references. Example: selected = 1, will select the first data pair from data_list to be plotted and compared with the references.
    xlim : list
        Limits for the x axis (2 theta).
    stretch_references : bool
        The plotted bars from the references that were imported from materialsproject database will color the selected plot for each value.
    stretch_thresh : int
        The colorizing of the selected plot based on the references values is extended to the sides with this argument.
    
    Returns
    -------
    None. However, since the variable patterns is global, it will append any result it finds to patterns.
    
    Example
    -----
    XRD_PLOTTER(['E:/Work/PhD/Data/Particles/Bi/Bi(NO3)3-DDCT-mwave-185-6min10s/XRD/DJ.csv'], [], ["Bi"], criteria_add = {'nsites': {'$lt': 10000}}, data_labels = ["Bi(NO3)3-DDCT-mwave-185-6min10s"])
    """
    dif = (xlim[1] - xlim[0])*1.05
    
    global patterns
    
    if (not stack_patterns and len(patterns) > 1) or len(patterns) == 0:
        del(patterns)
        gc.collect()
        patterns = [["Material id", "Formula", "Spacegroup", "Crystal system", "X-ray diffraction pattern", "Band gap/eV", "Final energy/eV", "Density (atomic)", "Density/(g/cm3)", "Metal's atomic weight/(g/mol)"]]
    
    if element_criteria != None and type(element_criteria) is list:
        criteria = dict({"elements":{"$all":element_criteria},"nelements":len(element_criteria)}, **criteria_add if criteria_add != None else {})
        properties = ["material_id", "pretty_formula", "spacegroup", u"final_energy", "density_atomic", "density"]
        data = api_key.query(criteria=criteria, properties=properties)
        for i in data:
            material_id_list.append([i["material_id"], i["pretty_formula"], i["spacegroup"]["crystal_system"], i[u"final_energy"], i["density_atomic"], i["density"]])
    material_id_list = [i[0] for i in sorted(material_id_list, key=itemgetter(1, 2))]
    
    
    data_color        = [0.5*(i/len(data_list)) if (i != 0 and i != len(data_list)-1)
                         else 0 if i == 0
                         else 0.5 if i == len(data_list)-1
                         else None
                         for i in range(len(data_list))]
    
    material_id_color = [0.5*(i/len(material_id_list)) if (i != 0 and i != len(material_id_list)-1)
                         else 0 if i == 0
                         else 0.5 if i == len(material_id_list)-1
                         else None
                         for i in range(len(material_id_list))]
    
    h = ((-0.212+axis_height)/25.4)/0.75
    h0 = (len(material_id_list)+len(data_list))
    figsize=[((-0.212+axis_width)/25.4)/0.75,
             h0*h+(15/25.4)/0.75] # [W, H]
    ws = 1/0.4
    figsize = [ws*figsize[0], figsize[1]]
    fig = plt.figure(figsize=(figsize[0], figsize[1]))
    
    wscale = 1/ws
    hscale = h/figsize[1]
    #1/(h0+2*(15/25.4)/0.75)
    axs = []
    for i in range(h0):
        wpos   = (15/25.4)/(figsize[0])
        hpos   = (15/25.4)/(figsize[1])+i*hscale
        ax = fig.add_axes([wpos,hpos,
                           wscale,hscale])
        ax.set_yticks([])
        ax.set_xticks([]) if i != 0 else ax.set_xticks([i*10 for i in range(int(xlim[0]/10), int(xlim[1]/10)+1)])
        ax.set_xlim(xlim)
        ax.set_ylim([0,1])
        if i == 0:
            ax.set_xlabel(u'2θ/' + u'°')
        axs.append(ax)
    
    
    for i in range(len(data_list)):
        try:
            if os.path.isfile(data_list[i]):
                data_list[i] = read_csv(data_list[i])
        except:
            print("Error while reading csv file: "+str(data_list[i]))
        y = normalize(data_list[i][1])
        
        axs[i].plot(data_list[i][0], normalize(data_list[i][1]), color=[data_color[i], data_color[i], data_color[i]])
        axs[i].annotate(text = data_labels[i], xy = (dif, 0.5), ha="right")
        axs[selected-1].annotate(text = "(Selected)", xy = (dif, 0.35), ha="right")
    
    
    sx = data_list[selected-1][0]
    sy = normalize(data_list[selected-1][1])
    
    
    annotate_x = xlim[1]*1.02
    annotate_position = "left"
    if len(material_id_list) > 0:
        for m in range(len(material_id_list)):
            if m == 0:
                printProgressBar(0, len(material_id_list))
            mm = m + len(data_list)
            if ".json" not in material_id_list[m]:
                calc = XRDCalculator()
                connect = 0
                tries = 1
                while connect == 0 and tries < 5:
                    try:
                        l = api_key.query(criteria={"material_id": material_id_list[m]}, properties=["pretty_formula", "spacegroup", "band_gap", "tags", u"final_energy", "density_atomic", "density"])[0]
                        structure = api_key.get_structure_by_material_id(material_id_list[m])
                        formula = l["pretty_formula"]
                        spacegroup = l["spacegroup"]["symbol"]
                        crystal = l["spacegroup"]["crystal_system"]
                        energy = l[u"final_energy"]
                        density_atomic = l["density_atomic"]
                        density = l["density"]
                        pattern = calc.get_pattern(structure)
                        tags = l["tags"]
                        x = [pattern.x[i] for i in range(len(pattern.x)) if pattern.x[i] <= xlim[1] and pattern.x[i] >= xlim[0]]
                        y = [pattern.y[i] for i in range(len(pattern.x)) if pattern.x[i] <= xlim[1] and pattern.x[i] >= xlim[0]]
                        y = normalize(y)
                        plt.rcParams.update({'font.size': 7/0.75})
                        axs[mm].bar(x, np.ones(len(y))-y*0.5, bottom = y*0.5,
                                    #color=[material_id_color[m], material_id_color[m], material_id_color[m]],
                                    color=[0.99, 0, 0, 0.15],
                                    width = 0.5)
                        if stretch_references:
                            if selected != None:
                                colors = np.zeros(len(sx))
                                for i in x:
                                    if min(sx[stretch_thresh:]) < i < max(sx[:len(sx)-stretch_thresh-1]):
                                        for j in range(len(sx)):
                                            if sx[int(max(j-stretch_thresh, 0))] <= i <= sx[int(min(j+stretch_thresh,len(sx)-1))]:
                                                size = len(colors[int(max(j-stretch_thresh, 0)):int(min(j+stretch_thresh,len(sx)))+1])
                                                colors[int(max(j-stretch_thresh, 0)):int(min(j+stretch_thresh,len(sx)))+1] = np.ones(size)
                                colors = [[i-0.02 if i == 1 else 0,0,0] for i in colors]
                                axs[mm].scatter(sx,sy*0.5+0.5, c=colors, s=1, zorder=1)
                        axs[mm].plot(sx,sy*0.5+0.5, "k", zorder=0)
                        axs[mm].bar(x, y*0.5, bottom = 0,
                                    #color=[material_id_color[m], material_id_color[m], material_id_color[m]],
                                    color=[0.99, 0, 0, 1],
                                    width = 0.5)
                        axs[mm].annotate(text = material_id_list[m], xy = (annotate_x,
                                                                           0.90), ha=annotate_position, annotation_clip=False)
                        axs[mm].annotate(text = "".join(["$_"+i+"$" if i.isdigit() else i for i in formula]), xy = (annotate_x,
                    0.80), ha=annotate_position, annotation_clip=False)
                        axs[mm].annotate(text = spacegroup, xy = (annotate_x,
                                                                  0.70), ha=annotate_position, annotation_clip=False)
                        axs[mm].annotate(text = crystal, xy = (annotate_x,
                                                               0.60), ha=annotate_position, annotation_clip=False)
                        axs[mm].annotate(text = "Band gap: "+ str(int(l["band_gap"]*100)/100), xy = (annotate_x,
                                                               0.50), ha=annotate_position, annotation_clip=False)
                        plt.rcParams.update({'font.size': 6/0.75})
                        for i in range(0, len(tags)):
                            axs[mm].annotate(text = tags[i], xy = (annotate_x,
                                                                   0.42-0.5*i*0.15), ha=annotate_position, annotation_clip=False)
                        plt.rcParams.update({'font.size': 10.9/0.75})
                        if material_id_list[m] not in [z[0] for z in patterns]:
                            patterns.append([material_id_list[m], formula, spacegroup, crystal, np.array((x, y)).T, l["band_gap"], energy, density_atomic, density, metal_atomic_weight(formula)])
                        connect = 1
                    except Exception as e:
                        print("Connection error. Might not being able to find the material with the id: " + material_id_list[m])
                        print(e)
                        tries += 1
                        time.sleep(1)
            else:
                id_mp = os.path.basename(material_id_list[m]).split("_")[0]
                url = "https://materialsproject.org/materials/"+id_mp
                title = [t.get_text() for t in BeautifulSoup(get(url).text, 'html.parser').find_all('title')][0]
                formula = title.split(" ")[2]
                spacegroup = title.split(" ")[4][:-1]
                pattern = XRD_JSON(material_id_list[m])
                y = [0 if pattern[0][i] < xlim[0] or pattern[0][i] > xlim[1] else pattern[1][i] for i in range(len(pattern[0]))]
                y = normalize(y)
                plt.bar(pattern[0], y, bottom = m, color=[material_id_color[m], material_id_color[m], material_id_color[m]], width = 0.5)
                plt.annotate(text = id_mp, xy = (dif, (m+1) - 0.15), ha="right")
                plt.annotate(text = "".join(["$_"+i+"$" if i.isdigit() else i for i in formula]), xy = (dif, (m+1) - 0.30), ha="right")
                plt.annotate(text = spacegroup, xy = (dif, (m+1) - 0.45), ha="right")
                patterns.append([material_id_list[m], formula, spacegroup, np.array((x, y)).T, None])
            
            printProgressBar(m+1, len(material_id_list))



def grab_information(element_criteria = ["Fe", "S"], criteria_add = {'nsites': {'$lt': 1000000}, "theoretical": False}, xlim = [10, 90]):
    """
    Description
    -------
    Same as XRD_PLOTTER but simply fetches the patterns from materialsproject wihtout plotting anything.
    """
    
    criteria = dict({"elements":{"$all":element_criteria},"nelements":len(element_criteria)}, **criteria_add if criteria_add != None else {})
    properties = ["material_id", "pretty_formula", "spacegroup", u"final_energy", "density_atomic", "density"]
    global patterns
    material_id_list = []
    data = api_key.query(criteria=criteria, properties=properties)
    for i in data:
        material_id_list.append([i["material_id"], i["pretty_formula"], i["spacegroup"]["crystal_system"], i[u"final_energy"], i["density_atomic"], i["density"]])
    material_id_list = [i[0] for i in sorted(material_id_list, key=itemgetter(1, 2))]
    for m in range(len(material_id_list)):
        calc = XRDCalculator()
        connect = 0
        tries = 1
        while connect == 0 and tries < 5:
            try:
                l = api_key.query(criteria={"material_id": material_id_list[m]}, properties=["pretty_formula", "spacegroup", "band_gap", "tags", u"final_energy", "density_atomic", "density"])[0]
                structure = api_key.get_structure_by_material_id(material_id_list[m])
                formula = l["pretty_formula"]
                spacegroup = l["spacegroup"]["symbol"]
                crystal = l["spacegroup"]["crystal_system"]
                energy = l[u"final_energy"]
                density_atomic = l["density_atomic"]
                density = l["density"]
                pattern = calc.get_pattern(structure)
                x = [pattern.x[i] for i in range(len(pattern.x)) if pattern.x[i] <= xlim[1] and pattern.x[i] >= xlim[0]]
                y = [pattern.y[i] for i in range(len(pattern.x)) if pattern.x[i] <= xlim[1] and pattern.x[i] >= xlim[0]]
                y = normalize(y)
                if material_id_list[m] not in [z[0] for z in patterns]:
                    patterns.append([material_id_list[m], formula, spacegroup, crystal, np.array((x, y)).T, l["band_gap"], energy, density_atomic, density, metal_atomic_weight(formula)])
                connect = 1
                printProgressBar(m+1, len(material_id_list))
            except Exception as e:
                print("Connection error. Might not being able to find the material with the id: " + material_id_list[m])
                print(e)
                tries += 1
                time.sleep(1)


# Auxiliary functions:

def XRD_JSON(file, data_only = True):
    with open(file, 'r') as outfile:
        dic = loads(outfile.read())
    if data_only:
        return [i[2] for i in dic["pattern"]], [i[0] for i in dic["pattern"]]
    else:
        return dic

def read_csv(file_name : str, skip : str = '[Scan points]'):
    with open(file_name, "r") as f_in:
        lines = [line for line in f_in.readlines() if line != []]
    # with open(file_name, "w") as f_in:
    #     f_in.writelines(lines)
    #     f_in.close()
    skiprows = [i for i in range(len(lines)) if skip in lines[i]][0]+1
    data = pd.read_csv(file_name, skiprows=skiprows)
    x = np.array(data["Angle"])
    y = np.array(data["Intensity"])
    return (x, y)

def metal_atomic_weight(formula : str):
    d = {'Ag': 107.868, 'Al': 26.98154, 'As': 74.9216, 'Au': 196.9665, 'B': 10.81, 'Ba': 137.33, 'Be': 9.01218, 'Bi': 208.9804, 'Ca': 40.08, 'Cd': 112.41, 'Ce': 140.12, 'Co': 58.9332, 'Cr': 51.996, 'Cs': 132.9054, 'Cu': 63.546, 'Dy': 162.5, 'Er': 167.26, 'Eu': 151.96, 'Fe': 55.847, 'Ga': 69.72, 'Gd': 157.25, 'Ge': 72.59, 'Hf': 178.49, 'Ho': 164.9304, 'In': 114.82, 'Ir': 192.22, 'K': 39.0983, 'La': 138.9055, 'Li': 6.941, 'Lu': 174.967, 'Mg': 24.305, 'Mn': 54.938, 'Mo': 95.94, 'Na': 22.98977, 'Nb': 92.9064, 'Nd': 144.24, 'Ni': 58.7, 'Os': 190.2, 'Pb': 207.2, 'Pd': 106.4, 'Pr': 140.9077, 'Pt': 195.09, 'Rb': 85.4678, 'Re': 186.207, 'Rh': 102.9055, 'Ru': 101.07, 'Sb': 121.75, 'Sc': 44.9559, 'Si': 28.0855, 'Sm': 150.4, 'Sn': 118.69, 'Sr': 87.62, 'Ta': 180.9479, 'Tb': 158.9254, 'Te': 127.6, 'Ti': 47.9, 'Tl': 204.37, 'Tm': 168.9342, 'V': 50.9415, 'W': 183.85, 'Y': 88.9059, 'Yb': 173.04, 'Zn': 65.38, 'Zr': 91.22}
    try:
        return sum([d[x] for x in d.keys() if x in formula])
    except:
        return None


def normalize(array, between = [0,1]):
    return between[0]+(between[1] - between[0])*(array-np.min(array))/(np.max(array)-np.min(array))


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 10, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * ((iteration) / float(total)))
    filledLength = int(length * iteration // total) + 1
    bar = fill * filledLength + '-' * ((length) - filledLength)
    print('\r%s |%s| %s%% %s | %s/%s' % (prefix, bar, percent, suffix, str(iteration), str(total)), end = printEnd)
    return None