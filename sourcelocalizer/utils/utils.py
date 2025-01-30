import os

def _fetch_labels(fname):
    fname_xml = os.path.splitext(fname)[0] + '.xml'
    if os.path.exists(fname_xml):
        import xml.etree.ElementTree as ET
        tree = ET.parse(fname_xml)
        root = tree.getroot()
        labels = []
        for label in root.find("data").findall("label"):
            index = label.find("index").text
            name = label.find("name").text
            labels.append((index, name))
        labels = {idx: name for idx, name in labels}
        return labels
    else:
        fname_txt = os.path.splitext(fname)[0] + '.txt'
        if os.path.exists(fname_txt):
            with open(fname_txt, 'r') as f:
                labels = f.readlines()
            labels = {str(idx): name.strip() for idx, name in enumerate(labels)}
            return labels
        else:
            return None