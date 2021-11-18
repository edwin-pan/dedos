import poppy
import os
import numpy as np

class ZernikeGenerator():
    def __init__(self, num_terms, save_path='./', patch_size=1356, scale_factor=1e-6):
        self.num_terms = num_terms
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.save_fname = os.path.join(save_path, 'zernike_volume_%d.npy'%self.patch_size)
        # Generating the zernike basis functions is expensive - if possible, only do it once.
        if not os.path.exists(self.save_fname):
            print(f"saving zernike volumn to {self.save_fname}")
            self.zernike_volume = self.compute_zernike_volume(resolution=self.patch_size, 
                                                              n_terms=self.num_terms, 
                                                              scale_factor=self.scale_factor).astype(np.float32)
            np.save(self.save_fname, self.zernike_volume)
        else:
            print(f"loading zernike volumn from {self.save_fname}")
            self.zernike_volume = np.load(self.save_fname)

    def compute_zernike_volume(self, resolution, n_terms, scale_factor):
        zernike_volume = poppy.zernike.zernike_basis(nterms=n_terms, npix=resolution, outside=0.0)
        return zernike_volume * scale_factor

