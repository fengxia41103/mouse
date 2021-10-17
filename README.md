1. Skeleton video: deeplabcut &rarr; .h5 &rarr; script &rarr; .mp4, in
   `/train_video` folder.
2. In `/train_data` folder, CSV files, three cols: start, end,
   duration. Hand labeled grooming moments.
3. Initialize/config model (**critical**, many magic numbers, all
   about tensorflow & keras), then train model, save to a `.h5`
4. Output prediction CSV
5. (optional) Our decision model: when to mark 1 or 0
6. plot based on decision output or prediction CSV
