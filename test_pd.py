import pandas as pd
""""
groups = ["Modern Web", "DevOps", "Cloud", "Big Data", "Security", "自我挑戰組"]
ironmen = [59, 9, 19, 14, 6, 77]

ironmen_dict = {
                "groups": groups,
                "ironmen": ironmen
}

# 建立 data frame
ironmen_df = pd.DataFrame(ironmen_dict)

# 刪除觀測值
ironmen_df_no_mw = ironmen_df.drop(0, axis = 0)
print(ironmen_df_no_mw)
print("---") # 分隔線

# 刪除欄位
ironmen_df_no_groups = ironmen_df.drop("groups", axis = 1)
print(ironmen_df_no_groups)
"""
