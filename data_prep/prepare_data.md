### Performing Alignment

- Calculating Camera Matrix:

  ```bash
  python data_prep/gen_camera_matrices.py --subset train
  python data_prep/gen_camera_matrices.py --subset test
  ```

  Which will produce the camera_matrices directory under ./bbfts_data/<train/test>

- Calculating Train set Homographies and find Pivot:

  ```bash
  python data_prep/gen_homographies.py --subset train
  ```

  Which will produce the homographies directory under ./bbfts_data/train.
  Also will print the Pivot sample and frame (e.g 'Pivot is 889.npy : 39')

- Calculating Test set Homographies using above pivot (e.g Sample 889, Frame 39):

   ```bash
  python data_prep/gen_homographies.py --subset test --pivot-name 889 --pivot-frame 39
  ```

  Which will produce the homographies directory under ./bbfts_data/test.

- Apply Homographies to align entire dataset:

  ```bash
  python data_prep/apply_homographies.py --subset train
  python data_prep/apply_homographies.py --subset test
  ```

  Which will produce the aligned_motion directory under ./bbfts_data/<train/test>.