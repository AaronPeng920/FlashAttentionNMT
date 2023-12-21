import json

with open('/home/app/project/ChineseNMT/data/json/test.json', 'r') as f:
    datas = json.load(f)
    m_len = 0
    for data in datas:
        eng, chn = data
        l = len(eng.split())
        if l > m_len:
            m_len = l
        
    print(len(datas), m_len)
    
    


