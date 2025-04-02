#Цільова змінна
y_column = ['Sleep_Quality']

#Колонки від яких залежить результат
x_columns = ['Age',
             'Gender_Female',
             'Gender_Male',
             'Gender_Other',
             'University_Year_1st Year',
             'University_Year_2nd Year',
             'University_Year_3rd Year',
             'University_Year_4th Year',
             'Sleep_Duration',
             'Study_Hours',
             'Screen_Time',
             'Caffeine_Intake',
             'Physical_Activity',
             'Weekday_Sleep_Start',
             'Weekend_Sleep_Start',
             'Weekday_Sleep_End',
             'Weekend_Sleep_End',
             ]

#Колонки, які мають текстові дані
columns_to_encode = ['Gender',
                     'University_Year']

columns_to_scale = ['Sleep_Quality',
                    'Age',
                    'Sleep_Duration',
                    'Study_Hours',
                    'Screen_Time',
                    'Caffeine_Intake',
                    'Physical_Activity',
                    'Weekday_Sleep_Start',
                    'Weekend_Sleep_Start',
                    'Weekday_Sleep_End',
                    'Weekend_Sleep_End'
                    ]

columns_to_split = ['Student_ID',
                    'Age',
                    'Gender',
                    'University_Year',
                    'Sleep_Duration',
                    'Study_Hours',
                    'Screen_Time',
                    'Caffeine_Intake',
                    'Physical_Activity',
                    'Weekday_Sleep_Start',
                    'Weekend_Sleep_Start',
                    'Weekend_Sleep_End',
                    'Weekday_Sleep_End']

