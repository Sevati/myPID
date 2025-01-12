import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_gen(inputFile):
    name=['fit','run','evtid','trk','pdg','x','y','z','numHits','fitSuccess','pval','chisq','ndf','X','Y','Z']
    df=pd.read_table(inputFile,names=name,sep=',',header=None)
    return df

def load_mcp(inputFile):
    name=['evtid','trkid','pdg','px','py','pz','pt','p']
    df=pd.read_table(inputFile,names=name,sep=',',header=None)
    return df

def load_hit(inputFile):
    name=['evtid','trkid','layer','wire','x','y','rt','tdc']
    df=pd.read_table(inputFile,names=name,sep=',',header=None)
    return df

def match(df, eff_mcp,hit):
    # 使用 merge 来直接找到匹配的行
    unique_hit = hit[['evtid', 'trkid']].drop_duplicates()
    alltrack = pd.merge(eff_mcp[['evtid', 'trkid','pdg','pt']], unique_hit[['evtid', 'trkid']], on=['evtid', 'trkid'], how='inner')    
    
    # #验证所有 alltrack 的 (evtid, pdg) 组合都是唯一的
    # duplicate_combinations = alltrack[['evtid', 'pdg']].duplicated(keep=False)
    # duplicates = alltrack[duplicate_combinations]
    # # 输出重复的组合和数量
    # if not duplicates.empty:
    #     print(f"有 {len(duplicates)} 行重复的 (evtid, pdg) 组合")
    # else:
    #     print("所有 (evtid, pdg) 组合都是唯一的")

    # merged = pd.merge(alltrack[['evtid', 'trkid','pdg','pt']],df[['evtid','pdg','fitSuccess']], on=['evtid','pdg'], how='inner')

    df_aggregated = df.groupby(['evtid', 'pdg'], as_index=False)['fitSuccess'].max()
    fit_success_map = df_aggregated.set_index(['evtid', 'pdg'])['fitSuccess']
    alltrack['fitSuccess'] = alltrack.set_index(['evtid', 'pdg']).index.map(fit_success_map)

    # nan_count = alltrack['fitSuccess'].isna().sum()
    # print(f"NaN 值的个数: {nan_count}")
    # zero_count = (alltrack['fitSuccess'] == 0).sum()
    # print(f"值为 0 的个数: {zero_count}")
    # one_count = (alltrack['fitSuccess'] == 1).sum()
    # print(f"值为 1 的个数: {one_count}")

    return alltrack

def calculate_efficiencies_by_pt(index, alltrack, bins=14):
    bin_edges = np.linspace(alltrack['pt'].min(), alltrack['pt'].max(), bins+1)
    
    efficiencies = []
    bin_centers = []
    
    # 对每个 bin 进行计算
    for i in range(bins):
        # 获取每个 bin 中的 pt 范围
        pt_min = bin_edges[i]
        pt_max = bin_edges[i+1]
        
        # 获取该范围内的 mcp 数据
        mcp_bin = alltrack[(alltrack['pt'] >= pt_min) & (alltrack['pt'] < pt_max)]
        
        # 计算该 bin 的效率
        success = mcp_bin[mcp_bin['fitSuccess']==1]
        bin_efficiency = len(success) / len(mcp_bin) if len(mcp_bin) > 0 else 0
        efficiencies.append(bin_efficiency)
        bin_centers.append((pt_min + pt_max) / 2)  # 使用 bin 的中心值作为 x 值

    # 绘制效率图
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, efficiencies, marker='o', linestyle='-', color='b')
    plt.xlabel('pT (GeV/c)', fontsize=12)
    plt.ylabel('Efficiency', fontsize=12)
    plt.title('Efficiency vs. pT', fontsize=14)
    plt.grid(True)
    plt.savefig(f'/Users/Sevati/PycharmProjects/untitled/PID/pid_data/GENFIT_OUTPUT/my/{index}.jpg')  
    plt.close

    return bin_centers, efficiencies



if __name__ == "__main__":
    efficiencies,fadeeff = {},{}
    for i in range(2,8):
        inputFile = f'/Users/Sevati/PycharmProjects/untitled/PID/pid_data/GENFIT_OUTPUT/my/GenfitOut_{i}.txt'
        df = load_gen(inputFile)
        truthfile = f'/Users/Sevati/PycharmProjects/untitled/PID/pid_data/mcp/mcP_{i}.txt'
        mcp = load_mcp(truthfile)
        eff_mcp = mcp[(mcp['pdg'] == 211) | (mcp['pdg'] == -211)]
        hitfile = f'/Users/Sevati/PycharmProjects/untitled/PID/pid_data/2Ddata/hit_{i}.txt'
        hit = load_hit(hitfile)
        alltrack = match(df, eff_mcp, hit)

        bin_centers, efficiencies = calculate_efficiencies_by_pt(i, alltrack)

        success = alltrack[alltrack['fitSuccess']==1]
        # success = df[df['fitSuccess']==1]
        # alltrack = df[df['pdg']!=0]
        fadetrk = df[df['pdg']==0]
        efficiencies[i] = len(success)/len(alltrack)
        fadeeff[i] = len(fadetrk)/len(df)

    average_efficiency = sum(efficiencies.values()) / len(efficiencies)
    print(f"Average Efficiency = {average_efficiency:.4f}")
    average_fade_efficiency = sum(fadeeff.values()) / len(fadeeff)
    print(f"Average Fade Efficiency = {average_fade_efficiency:.4f}")

# f'/Users/Sevati/PycharmProjects/untitled/PID/pid_data/GENFIT_OUTPUT/xq/GenfitOut_{i}.txt'
# f'/Users/Sevati/PycharmProjects/untitled/PID/pid_data/GENFIT_OUTPUT/my/GenfitOut_{i}.txt'
