"""
models.py — Base model primitives for synthetic data generation.

Each function returns an ehtim.Image (single frame) or ehtim.Movie object.
All functions are pure: they accept explicit parameters and produce deterministic
outputs when given the same seed. No hidden globals.

Author: Rohan Dahale
"""

import os
import yaml
import numpy as np
import ehtim as eh
from ehtim.imaging.pol_imager_utils import qimage, uimage
from tqdm import tqdm
from typing import List


# ═══════════════════════════════════════════════════════════════════════
#  Parameter Validation
# ═══════════════════════════════════════════════════════════════════════

# Required parameters for each model type (keys that use params['key'], not params.get('key'))
_REQUIRED_PARAMS: dict = {
    'crescent':           ['fov_uas', 'npix', 'total_flux'],
    'ring':               ['fov_uas', 'npix', 'total_flux'],
    'disk':               ['fov_uas', 'npix', 'total_flux'],
    'edisk':              ['fov_uas', 'npix', 'total_flux'],
    'double':             ['fov_uas', 'npix', 'total_flux'],
    'point':              ['fov_uas', 'npix', 'total_flux'],
    'mring_hs':           ['fov_uas', 'npix', 'total_flux'],
    'mring_hs_cross':     ['fov_uas', 'npix', 'total_flux'],
    'mring_hs_not_center':['fov_uas', 'npix', 'total_flux'],
    'mring_hs_incoh':     ['fov_uas', 'npix', 'total_flux'],
    'mring_hs_pol':       ['fov_uas', 'npix', 'total_flux'],
    'varbeta2':           ['fov_uas', 'npix', 'total_flux'],
    'grmhd':              ['fov_uas', 'npix'],
    'grmhd_hs':           ['fov_uas', 'npix'],
}


def validate_params(model_type: str, params: dict) -> List[str]:
    """Check that required parameters are present for a model type.

    Args:
        model_type: registry key (e.g. 'mring_hs', 'crescent').
        params: merged parameter dict.

    Returns:
        List of missing key names (empty if all present).
    """
    required = _REQUIRED_PARAMS.get(model_type, [])
    return [k for k in required if k not in params]


# ═══════════════════════════════════════════════════════════════════════
#  Polarization Helpers
# ═══════════════════════════════════════════════════════════════════════

def _add_radial_polarization(im, linpol_frac, circpol_frac, qu_shift=np.pi/2):
    """Add radial EVPA pattern + circular polarization to an ehtim.Image.

    Args:
        im: ehtim.Image with I populated.
        linpol_frac: fractional linear polarization |m|.
        circpol_frac: fractional circular polarization |v|.
        qu_shift: phase offset between Q and U EVPA grids.

    Returns:
        Modified image (in-place).
    """
    xx = np.linspace(-1, 1, im.xdim)
    yy = np.linspace(-1, 1, im.ydim)
    xx, yy = np.meshgrid(xx, yy)
    angles = np.angle(xx + 1j * yy)

    im.qvec = qimage(im.imvec, linpol_frac * np.ones(im.imvec.shape),
                      angles.flatten() + qu_shift / 2)
    im.uvec = uimage(im.imvec, linpol_frac * np.ones(im.imvec.shape),
                      angles.flatten() - qu_shift / 2)
    im.vvec = circpol_frac * im.imvec
    return im


def _add_azimuthal_polarization(im, linpol_frac, circpol_frac):
    """Add azimuthal EVPA pattern (for disk model)."""
    ny, nx = im.ydim, im.xdim
    y, x = np.indices((ny, nx))
    x_center, y_center = nx // 2, ny // 2
    theta = np.arctan2(y - y_center, x - x_center)
    theta = np.mod(theta, 2 * np.pi)

    im.qvec = qimage(im.imvec, linpol_frac * np.ones(im.imvec.shape),
                      -theta.flatten())
    im.uvec = uimage(im.imvec, linpol_frac * np.ones(im.imvec.shape),
                      -theta.flatten())
    im.vvec = circpol_frac * im.imvec
    return im


# ═══════════════════════════════════════════════════════════════════════
#  Static Model Frame Builders
# ═══════════════════════════════════════════════════════════════════════

def make_crescent_frame(params: dict, obs_ref) -> 'eh.image.Image':
    """Create a single crescent (thick m-ring order=1) frame.

    Required keys in params:
        fov_uas, npix, total_flux, PA, beta1_abs, diameter_uas, alpha_uas,
        linpol_frac, circpol_frac
    """
    fov = params['fov_uas'] * eh.RADPERUAS
    npix = params['npix']
    PA = params.get('PA', 120)
    qu_shift = params.get('QU_shift', np.pi / 2)

    crescent_par = {
        'F0': params['total_flux'],
        'd': params.get('diameter_uas', 52) * eh.RADPERUAS,
        'x0': 0,
        'y0': 0,
        'beta_list': [params.get('beta1_abs', 0.23) *
                      np.exp(-1j * np.deg2rad(-PA))],
        'alpha': params.get('alpha_uas', 15) * eh.RADPERUAS,
    }

    model = eh.model.Model(ra=obs_ref.ra, dec=obs_ref.dec,
                            rf=obs_ref.rf, source=obs_ref.source)
    im = model.add_thick_mring(**crescent_par).make_image(fov, npix)

    linpol = params.get('linpol_frac', 0.2)
    circpol = params.get('circpol_frac', 0.002)
    _add_radial_polarization(im, linpol, circpol, qu_shift)
    return im


def make_ring_frame(params: dict, obs_ref) -> 'eh.image.Image':
    """Create a single ring (thick ring, symmetric) frame."""
    fov = params['fov_uas'] * eh.RADPERUAS
    npix = params['npix']
    qu_shift = params.get('QU_shift', np.pi / 2)

    ring_par = {
        'F0': params['total_flux'],
        'd': params.get('diameter_uas', 52) * eh.RADPERUAS,
        'alpha': params.get('alpha_uas', 15) * eh.RADPERUAS,
        'x0': 0,
        'y0': 0,
    }

    model = eh.model.Model(ra=obs_ref.ra, dec=obs_ref.dec,
                            rf=obs_ref.rf, source=obs_ref.source)
    im = model.add_thick_ring(**ring_par).make_image(fov, npix)

    linpol = params.get('linpol_frac', 0.2)
    circpol = params.get('circpol_frac', 0.002)
    # Ring EVPA convention: Q offset = qu_shift, U offset = 0.
    # Intentionally different from _add_radial_polarization (Q=+qu_shift/2, U=-qu_shift/2)
    # to give argbeta2=0 (purely radial) when qu_shift=pi/2.
    xx = np.linspace(-1, 1, im.xdim)
    yy = np.linspace(-1, 1, im.ydim)
    xx, yy = np.meshgrid(xx, yy)
    angles = np.angle(xx + 1j * yy)

    im.qvec = qimage(im.imvec, linpol * np.ones(im.imvec.shape),
                      angles.flatten() + qu_shift)
    im.uvec = uimage(im.imvec, linpol * np.ones(im.imvec.shape),
                      angles.flatten())
    im.vvec = circpol * im.imvec
    return im


def make_disk_frame(params: dict, obs_ref) -> 'eh.image.Image':
    """Create a single disk frame with azimuthal polarization."""
    fov = params['fov_uas'] * eh.RADPERUAS
    npix = params['npix']

    disk_par = {
        'F0': params['total_flux'],
        'd': params.get('disk_diameter_uas', 70) * eh.RADPERUAS,
        'x0': 0,
        'y0': 0,
    }

    model = eh.model.Model(ra=obs_ref.ra, dec=obs_ref.dec,
                            rf=obs_ref.rf, source=obs_ref.source)
    im = model.add_disk(**disk_par).make_image(fov, npix)

    linpol = params.get('linpol_frac', 0.2)
    circpol = params.get('circpol_frac', 0.002)
    _add_azimuthal_polarization(im, linpol, circpol)

    # Blur to smooth the sharp disk edge
    blur_uas = params.get('blur_uas', 15)
    im = im.blur_circ(fwhm_i=blur_uas * eh.RADPERUAS,
                      fwhm_pol=blur_uas * eh.RADPERUAS)
    return im


def make_edisk_frame(params: dict, obs_ref) -> 'eh.image.Image':
    """Create an eccentric (stretched) disk frame."""
    fov = params['fov_uas'] * eh.RADPERUAS
    npix = params['npix']
    dradius = params.get('dradius', 40)
    blur = params.get('blur_uas', 10)
    pangle = params.get('pangle_deg', 70)
    asymmetry = params.get('asymmetry', 0.8)

    empty = eh.image.make_empty(npix=npix, fov=fov,
                                 ra=obs_ref.ra, dec=obs_ref.dec,
                                 rf=obs_ref.rf, source=obs_ref.source)

    # Build elliptical disk
    dx = empty.psize
    xuas = -(np.arange(npix) - npix / 2) * dx / eh.RADPERUAS
    yuas = (np.arange(npix) - npix / 2) * dx / eh.RADPERUAS
    xuas, yuas = np.meshgrid(xuas, yuas)
    rr = np.sqrt(xuas ** 2 + (yuas / asymmetry) ** 2)
    idx = rr <= dradius
    jnu = np.zeros_like(rr)
    jnu[idx] = 1

    img = empty.copy()
    img.imvec = jnu.reshape(npix * npix)
    img.imvec *= params['total_flux'] / img.total_flux()

    # Blur
    img = img.blur_circ(blur * eh.RADPERUAS)

    # Rotate
    img = img.rotate(np.deg2rad(pangle))

    # Polarization
    linpol = params.get('linpol_frac', 0.2)
    circpol = params.get('circpol_frac', 0.002)
    img.qvec = qimage(img.imvec, linpol * np.ones(img.imvec.shape),
                       -np.ones(len(img.imvec)) * np.pi / 6)
    img.uvec = uimage(img.imvec, linpol * np.ones(img.imvec.shape),
                       -np.ones(len(img.imvec)) * np.pi / 6)
    img.vvec = circpol * img.imvec
    return img


def make_double_frame(params: dict, obs_ref) -> 'eh.image.Image':
    """Create a double Gaussian frame."""
    fov = params['fov_uas'] * eh.RADPERUAS
    npix = params['npix']
    flux = params['total_flux']

    sep = params.get('separation_uas', 17.5) * eh.RADPERUAS
    fwhm1 = params.get('fwhm1_uas', 35) * eh.RADPERUAS
    fwhm2 = params.get('fwhm2_uas', 25) * eh.RADPERUAS
    frat1 = params.get('flux_ratio1', 0.35 / 0.6)

    d1_par = {
        'F0': flux * frat1,
        'FWHM_maj': fwhm1,
        'FWHM_min': fwhm1,
        'PA': 0.0,
        'x0': sep / 2,
        'y0': (sep + fwhm1) / 2,
    }
    d2_par = {
        'F0': flux * (1.0 - frat1),
        'FWHM_maj': fwhm2,
        'FWHM_min': fwhm2,
        'PA': 0.0,
        'x0': -sep / 2,
        'y0': -(sep + fwhm1) / 2,
    }

    model = eh.model.Model(ra=obs_ref.ra, dec=obs_ref.dec,
                            rf=obs_ref.rf, source=obs_ref.source)
    double1 = model.add_gauss(**d1_par).make_image(fov, npix)
    double2 = model.add_gauss(**d2_par).make_image(fov, npix)

    linpol = params.get('linpol_frac', 0.2)
    circpol = params.get('circpol_frac', 0.002)

    double1.qvec = qimage(double1.imvec, linpol * np.ones(double1.imvec.shape),
                           np.ones(len(double1.imvec)) * np.pi / 4)
    double1.uvec = uimage(double1.imvec, linpol * np.ones(double1.imvec.shape),
                           np.ones(len(double1.imvec)) * np.pi / 4)
    double1.vvec = circpol * double1.imvec

    double2.qvec = qimage(double2.imvec, linpol * np.ones(double2.imvec.shape),
                           -np.ones(len(double2.imvec)) * np.pi / 4)
    double2.uvec = uimage(double2.imvec, linpol * np.ones(double2.imvec.shape),
                           -np.ones(len(double2.imvec)) * np.pi / 4)
    double2.vvec = circpol * double2.imvec

    result = eh.image.make_empty(npix=npix, fov=fov,
                                  ra=obs_ref.ra, dec=obs_ref.dec,
                                  rf=obs_ref.rf, source=obs_ref.source)
    result.ivec = double1.ivec + double2.ivec
    result.qvec = double1.qvec + double2.qvec
    result.uvec = double1.uvec + double2.uvec
    result.vvec = double1.vvec + double2.vvec
    return result


def make_point_frame(params: dict, obs_ref) -> 'eh.image.Image':
    """Create a point-source (compact Gaussian + large disk) frame."""
    fov = params['fov_uas'] * eh.RADPERUAS
    npix = params['npix']
    flux = params['total_flux']

    fwhm_compact = params.get('compact_fwhm_uas', 25) * eh.RADPERUAS
    frat_compact = params.get('compact_flux_ratio', 0.15 / 0.6)
    disk_diam = params.get('disk_diameter_uas', 110) * eh.RADPERUAS
    blur_fwhm = params.get('blur_uas', 15) * eh.RADPERUAS

    # Compact Gaussian
    d1_par = {
        'F0': flux * frat_compact,
        'FWHM_maj': fwhm_compact,
        'FWHM_min': fwhm_compact,
        'PA': 0.0,
        'x0': 0,
        'y0': 0,
    }
    # Large disk
    disk_par = {
        'F0': flux * (1.0 - frat_compact),
        'd': disk_diam,
        'x0': 0,
        'y0': 0,
    }

    model = eh.model.Model(ra=obs_ref.ra, dec=obs_ref.dec,
                            rf=obs_ref.rf, source=obs_ref.source)
    disk = model.add_disk(**disk_par).make_image(fov, npix)
    disk = disk.blur_circ(fwhm_i=blur_fwhm, fwhm_pol=blur_fwhm)

    model2 = eh.model.Model(ra=obs_ref.ra, dec=obs_ref.dec,
                             rf=obs_ref.rf, source=obs_ref.source)
    gauss = model2.add_gauss(**d1_par).make_image(fov, npix)

    result = eh.image.make_empty(npix=npix, fov=fov,
                                  ra=obs_ref.ra, dec=obs_ref.dec,
                                  rf=obs_ref.rf, source=obs_ref.source)
    result.ivec = gauss.ivec + disk.ivec

    linpol = params.get('linpol_frac', 0.2)
    circpol = params.get('circpol_frac', 0.002)
    # Point EVPA convention: Q=angles+pi/2, U=angles (same as ring, argbeta2=0).
    # Intentionally different from _add_radial_polarization (Q=+qu_shift/2, U=-qu_shift/2).
    xx = np.linspace(-1, 1, result.xdim)
    yy = np.linspace(-1, 1, result.ydim)
    xx, yy = np.meshgrid(xx, yy)
    angles = np.angle(xx + 1j * yy)

    result.qvec = qimage(result.imvec, linpol * np.ones(result.imvec.shape),
                          angles.flatten() + np.pi / 2)
    result.uvec = uimage(result.imvec, linpol * np.ones(result.imvec.shape),
                          angles.flatten())
    result.vvec = circpol * result.imvec
    return result


# ═══════════════════════════════════════════════════════════════════════
#  Dynamic Model Frame Builders (mring + hotspot variants)
# ═══════════════════════════════════════════════════════════════════════

def make_mring_hs_frame(angle, params: dict, obs_ref) -> 'eh.image.Image':
    """Create a single mring + orbiting Gaussian hotspot frame.

    Args:
        angle: orbital phase angle (radians).
        params: dict with keys:
            total_flux (Jy), hs_flux (Jy),
            PA (deg) — mring brightness asymmetry position angle,
            diameter_uas, alpha_uas, beta1_abs — mring geometry,
            ring_radius_uas — hotspot orbital radius,
            hs_fwhm_uas — hotspot Gaussian FWHM,
            hs_x_offset, hs_y_offset (uas) — offset from ring center (optional),
            fov_uas, npix — image grid,
            linpol_frac — fractional linear polarization (|m|),
            circpol_frac — fractional circular polarization (|v|),
            QU_shift (rad) — global EVPA twist of the mring polarization:
                π/2 (default) = radial pattern (tangential B-field),
                0             = tangential pattern (radial B-field),
                any value     = global offset applied to the radial pattern.
    """
    fov = params['fov_uas'] * eh.RADPERUAS
    npix = params['npix']
    PA = params.get('PA', 120)
    qu_shift = params.get('QU_shift', np.pi / 2)
    hs_flux = params.get('hs_flux', 0.3)
    mring_flux = params.get('total_flux', 2.7) - hs_flux

    crescent_par = {
        'F0': mring_flux,
        'd': params.get('diameter_uas', 52) * eh.RADPERUAS,
        'x0': 0,
        'y0': 0,
        'beta_list': [params.get('beta1_abs', 0.23) *
                      np.exp(-1j * np.deg2rad(-PA))],
        'alpha': params.get('alpha_uas', 15) * eh.RADPERUAS,
    }

    model = eh.model.Model(ra=obs_ref.ra, dec=obs_ref.dec,
                            rf=obs_ref.rf, source=obs_ref.source)
    im = model.add_thick_mring(**crescent_par).make_image(fov, npix)

    linpol = params.get('linpol_frac', 0.2)
    circpol = params.get('circpol_frac', 0.002)
    _add_radial_polarization(im, linpol, circpol, qu_shift)

    # Hotspot position
    R = params.get('ring_radius_uas', 26) * eh.RADPERUAS
    xg = R * np.cos(angle) + params.get('hs_x_offset', 0) * eh.RADPERUAS
    yg = R * np.sin(angle) + params.get('hs_y_offset', 0) * eh.RADPERUAS
    hs_fwhm = params.get('hs_fwhm_uas', 20) * eh.RADPERUAS

    gauss_par = {
        'flux': hs_flux,
        'beamparams': [hs_fwhm, hs_fwhm, 0, xg, yg],
    }
    im = im.add_gauss(**gauss_par, pol=None)
    return im


def make_mring_hs_pol_frame(angle, params: dict, obs_ref) -> 'eh.image.Image':
    """Create a single mring + polarized orbiting hotspot frame.

    The crescent has lower intrinsic polarization (5%); the hotspot
    carries its own pol with EVPA rotating with orbital angle.
    """
    fov = params['fov_uas'] * eh.RADPERUAS
    npix = params['npix']
    PA = params.get('PA', 120)
    hs_flux = params.get('hs_flux', 0.3)
    mring_flux = params.get('total_flux', 2.7) - hs_flux

    crescent_par = {
        'F0': mring_flux,
        'd': params.get('diameter_uas', 52) * eh.RADPERUAS,
        'x0': 0,
        'y0': 0,
        'beta_list': [params.get('beta1_abs', 0.23) *
                      np.exp(-1j * np.deg2rad(-PA))],
        'alpha': params.get('alpha_uas', 15) * eh.RADPERUAS,
    }

    model = eh.model.Model(ra=obs_ref.ra, dec=obs_ref.dec,
                            rf=obs_ref.rf, source=obs_ref.source)
    im = model.add_thick_mring(**crescent_par).make_image(fov, npix)

    # Low polarization on crescent
    crescent_linpol = params.get('crescent_linpol', 0.05)
    crescent_circpol = params.get('crescent_circpol', 0.001)

    xx = np.linspace(-1, 1, im.xdim)
    yy = np.linspace(-1, 1, im.ydim)
    xx, yy = np.meshgrid(xx, yy)
    angles_grid = np.angle(xx + 1j * yy)

    im.qvec = qimage(im.imvec, crescent_linpol * np.ones(im.imvec.shape),
                      angles_grid.flatten() + np.pi / 2)
    im.uvec = uimage(im.imvec, crescent_linpol * np.ones(im.imvec.shape),
                      angles_grid.flatten())
    im.vvec = crescent_circpol * im.imvec

    # Create polarized hotspot as a separate image
    hs_fwhm = params.get('hs_fwhm_uas', 20) * eh.RADPERUAS
    gauss_par = {
        'flux': hs_flux,
        'beamparams': [hs_fwhm, hs_fwhm, 0, 0, 0],
    }
    gauss_im = eh.image.make_empty(npix=npix, fov=fov,
                                    ra=obs_ref.ra, dec=obs_ref.dec,
                                    rf=obs_ref.rf, source=obs_ref.source)
    gauss_im = gauss_im.add_gauss(**gauss_par, pol=None)

    # Add rotating EVPA to hotspot
    hs_linpol = params.get('hs_linpol', 0.2)
    hs_circpol = params.get('hs_circpol', 0.002)
    gauss_im.qvec = qimage(gauss_im.imvec,
                            hs_linpol * np.ones(gauss_im.imvec.shape),
                            np.zeros(len(gauss_im.imvec)) - angle)
    gauss_im.uvec = uimage(gauss_im.imvec,
                            hs_linpol * np.ones(gauss_im.imvec.shape),
                            np.zeros(len(gauss_im.imvec)) - angle)
    gauss_im.vvec = hs_circpol * gauss_im.imvec

    # Shift hotspot to orbital position
    R = params.get('ring_radius_uas', 26) * eh.RADPERUAS
    xg = R * np.cos(angle)
    yg = R * np.sin(angle)
    gauss_im = gauss_im.shift_fft([xg, yg])

    # Combine
    im.imvec += gauss_im.imvec
    im.qvec += gauss_im.qvec
    im.uvec += gauss_im.uvec
    im.vvec += gauss_im.vvec
    return im


def make_mring_hs_cross_frame(x_pos, params: dict, obs_ref) -> 'eh.image.Image':
    """Create a frame with mring + linearly-crossing Gaussian hotspot.

    Args:
        x_pos: x-position of the hotspot (in radians, not µas).
    """
    fov = params['fov_uas'] * eh.RADPERUAS
    npix = params['npix']
    PA = params.get('PA', 120)
    qu_shift = params.get('QU_shift', np.pi / 2)
    hs_flux = params.get('hs_flux', 0.3)
    mring_flux = params.get('total_flux', 2.7) - hs_flux

    crescent_par = {
        'F0': mring_flux,
        'd': params.get('diameter_uas', 52) * eh.RADPERUAS,
        'x0': 0,
        'y0': 0,
        'beta_list': [params.get('beta1_abs', 0.23) *
                      np.exp(-1j * np.deg2rad(-PA))],
        'alpha': params.get('alpha_uas', 15) * eh.RADPERUAS,
    }

    model = eh.model.Model(ra=obs_ref.ra, dec=obs_ref.dec,
                            rf=obs_ref.rf, source=obs_ref.source)
    im = model.add_thick_mring(**crescent_par).make_image(fov, npix)

    linpol = params.get('linpol_frac', 0.2)
    circpol = params.get('circpol_frac', 0.002)
    _add_radial_polarization(im, linpol, circpol, qu_shift)

    hs_fwhm = params.get('hs_fwhm_uas', 20) * eh.RADPERUAS
    gauss_par = {
        'flux': hs_flux,
        'beamparams': [hs_fwhm, hs_fwhm, 0, x_pos, 0],
    }
    im = im.add_gauss(**gauss_par, pol=None)
    return im


def make_varbeta2_frame(t_elapsed, params: dict, obs_ref) -> 'eh.image.Image':
    """Create a frame with rotating EVPA pattern (varying beta2).

    Args:
        t_elapsed: time elapsed from start (hours).
    """
    fov = params['fov_uas'] * eh.RADPERUAS
    npix = params['npix']
    PA = params.get('PA', 120)
    T = params.get('varbeta_period_hr', 1.3333)

    crescent_par = {
        'F0': params.get('total_flux', 2.7),
        'd': params.get('diameter_uas', 52) * eh.RADPERUAS,
        'x0': 0,
        'y0': 0,
        'beta_list': [params.get('beta1_abs', 0.23) *
                      np.exp(-1j * np.deg2rad(-PA))],
        'alpha': params.get('alpha_uas', 15) * eh.RADPERUAS,
    }

    model = eh.model.Model(ra=obs_ref.ra, dec=obs_ref.dec,
                            rf=obs_ref.rf, source=obs_ref.source)
    im = model.add_thick_mring(**crescent_par).make_image(fov, npix)

    # Rotating vector field
    xx = np.linspace(-1, 1, npix)
    yy = np.linspace(-1, 1, npix)
    xx, yy = np.meshgrid(xx, yy)
    r = np.sqrt(xx**2 + yy**2)

    theta_rot = -2 * np.pi * t_elapsed / (2 * T)
    radial_x = xx / (r + 1e-6)
    radial_y = yy / (r + 1e-6)
    rotated_x = radial_x * np.cos(theta_rot) - radial_y * np.sin(theta_rot)
    rotated_y = radial_x * np.sin(theta_rot) + radial_y * np.cos(theta_rot)
    rotated_norm = np.sqrt(rotated_x**2 + rotated_y**2)
    rotated_x /= rotated_norm
    rotated_y /= rotated_norm
    chi = np.arctan2(rotated_y, rotated_x)

    linpol = params.get('linpol_frac', 0.2)
    circpol = params.get('circpol_frac', 0.002)

    stokes_I = im.imarr(pol='I')
    im.ivec = stokes_I.flatten()
    im.qvec = (stokes_I * linpol * np.cos(2 * chi)).flatten()
    im.uvec = -(stokes_I * linpol * np.sin(2 * chi)).flatten()  # ehtim sign convention
    im.vvec = circpol * im.ivec
    return im


# ═══════════════════════════════════════════════════════════════════════
#  Movie Builders (assemble frames into eh.Movie)
# ═══════════════════════════════════════════════════════════════════════

def _get_frame_times(obs_ref, t_gather=30):
    """Compute movie frame times and count from observation reference."""
    tstart = obs_ref.data['time'][0]
    tstop = obs_ref.data['time'][-1]
    obs_ref.tstart = tstart
    obs_ref.tstop = tstop
    Nframes = len(obs_ref.split_obs(t_gather=t_gather))
    times = np.linspace(tstart, tstop, Nframes)
    return times, Nframes


def _compute_orbital_angles(times, period_min, direction='CW'):
    """Compute orbital angles for a hotspot.

    Args:
        times: array of frame times.
        period_min: orbital period in minutes.
        direction: 'CW' (clockwise) or 'CCW' (counter-clockwise).

    Returns:
        array of angles in radians.
    """
    period_hr = period_min / 60.0
    Nloops = (times[-1] - times[0]) / period_hr
    sign = 1 if direction == 'CW' else -1
    angles = np.linspace(0, sign, len(times)) * 2 * np.pi * Nloops
    return angles


def make_static_movie(frame_func, params: dict, obs_ref) -> 'eh.movie.Movie':
    """Create a movie from a static frame repeated at each time step.

    Args:
        frame_func: callable(params, obs_ref) -> eh.Image
        params: parameter dict.
        obs_ref: reference observation.
    """
    times, Nframes = _get_frame_times(obs_ref)
    # Build the frame once and copy for each time step (avoids redundant computation)
    base_frame = frame_func(params, obs_ref)
    framelist = []
    for t in tqdm(times, desc="Building static movie"):
        frame = base_frame.copy()
        frame.time = t
        framelist.append(frame)
    return eh.movie.merge_im_list(framelist, interp='linear', bounds_error=True)


def make_mring_hs_movie(params: dict, obs_ref) -> 'eh.movie.Movie':
    """Create a movie of mring + orbiting hotspot.

    Extra params: period_min, direction ('CW'/'CCW').
    """
    times, Nframes = _get_frame_times(obs_ref)
    period_min = params.get('period_min', 80)
    direction = params.get('direction', 'CW')
    angles = _compute_orbital_angles(times, period_min, direction)

    framelist = []
    for angle, t in tqdm(zip(angles, times), total=len(times),
                          desc=f"Building mring+hs movie"):
        frame = make_mring_hs_frame(angle, params, obs_ref)
        frame.time = t
        framelist.append(frame)
    return eh.movie.merge_im_list(framelist, interp='linear', bounds_error=True)


def make_mring_hs_pol_movie(params: dict, obs_ref) -> 'eh.movie.Movie':
    """Create a movie of mring + polarized hotspot."""
    times, Nframes = _get_frame_times(obs_ref)
    period_min = params.get('period_min', 80)
    direction = params.get('direction', 'CW')
    angles = _compute_orbital_angles(times, period_min, direction)

    framelist = []
    for angle, t in tqdm(zip(angles, times), total=len(times),
                          desc="Building mring+hs-pol movie"):
        frame = make_mring_hs_pol_frame(angle, params, obs_ref)
        frame.time = t
        framelist.append(frame)
    return eh.movie.merge_im_list(framelist, interp='linear', bounds_error=True)


def make_mring_hs_cross_movie(params: dict, obs_ref) -> 'eh.movie.Movie':
    """Create a movie with crossing (back-and-forth) hotspot.

    The hotspot oscillates linearly from +x_range to -x_range and back,
    with the pattern tiled to fill the observation window.

    Extra params:
        hs_cross_range_uas: half-amplitude of oscillation (default 60 µas).
        period_min: full oscillation period in minutes (default 90).
    """
    times, Nframes = _get_frame_times(obs_ref)

    x_range = params.get('hs_cross_range_uas', 60.0) * eh.RADPERUAS
    period_min = params.get('period_min', 90.0)
    period_hr = period_min / 60.0

    # Generate one full oscillation cycle (+x_range → -x_range → +x_range)
    # sampled at the frame cadence
    dt_hr = (times[-1] - times[0]) / max(len(times) - 1, 1)
    frames_per_half = max(2, int(round((period_hr / 2.0) / dt_hr)))
    half_fwd = np.linspace(x_range, -x_range, frames_per_half, endpoint=False)
    half_bck = np.linspace(-x_range, x_range, frames_per_half, endpoint=False)
    one_cycle = np.concatenate([half_fwd, half_bck])

    # Tile the cycle to cover all frames
    n_cycles = int(np.ceil(Nframes / len(one_cycle)))
    x_pos = np.tile(one_cycle, n_cycles)[:Nframes]

    framelist = []
    for xp, t in tqdm(zip(x_pos, times), total=len(times),
                       desc="Building mring+hs-cross movie"):
        frame = make_mring_hs_cross_frame(xp, params, obs_ref)
        frame.time = t
        framelist.append(frame)
    return eh.movie.merge_im_list(framelist, interp='linear', bounds_error=True)


def make_mring_hs_incoh_movie(params: dict, obs_ref) -> 'eh.movie.Movie':
    """Create a movie with incoherent (randomly placed) hotspot.

    Uses the DAR (Distributed Angle Random) pattern from v3.
    """
    times, Nframes = _get_frame_times(obs_ref)

    t_boundary1 = params.get('besttime_start', 10.85)
    t_boundary2 = params.get('besttime_stop', 14.05)

    # Hotspot angular positions (radians) for each time period.
    # Chosen so that successive positions are non-adjacent on the ring,
    # making the hotspot appear to jump randomly (incoherent motion).
    # shift = 15 deg keeps positions off the cardinal axes.
    shift = 15 * np.pi / 180
    angles_p1 = [-2.44346095, -np.pi / 2 + shift]           # pre-besttime (~2 positions)
    angles_p2 = [-np.pi / 2 + shift, np.pi - shift, 0 + shift,
                 np.pi / 2 - shift, -3 * np.pi / 4 + shift, np.pi / 4 - shift]  # besttime (~6 positions cycling)
    angles_p3 = [np.pi / 4 - shift, 1.74532925]             # post-besttime (~2 positions)

    def distribute_angles(angle_list, num_frames):
        if not angle_list or num_frames == 0:
            return []
        n = len(angle_list)
        return [angle_list[i % n] for i in range(num_frames)]

    times_p1 = times[times < t_boundary1]
    times_p2 = times[(times >= t_boundary1) & (times <= t_boundary2)]
    times_p3 = times[times > t_boundary2]

    all_angles = np.concatenate([
        distribute_angles(angles_p1, len(times_p1)),
        distribute_angles(angles_p2, len(times_p2)),
        distribute_angles(angles_p3, len(times_p3)),
    ])
    all_times = np.concatenate([times_p1, times_p2, times_p3])

    framelist = []
    for angle, t in tqdm(zip(all_angles, all_times), total=len(all_times),
                          desc="Building mring+hs-incoh movie"):
        frame = make_mring_hs_frame(angle, params, obs_ref)
        frame.time = t
        framelist.append(frame)
    return eh.movie.merge_im_list(framelist, interp='linear', bounds_error=True)


def make_mring_hs_not_center_movie(params: dict, obs_ref) -> 'eh.movie.Movie':
    """Create a movie with hotspot orbiting off-center (offset from ring)."""
    times, Nframes = _get_frame_times(obs_ref)
    period_min = params.get('period_min', 180)
    direction = params.get('direction', 'CW')
    angles = _compute_orbital_angles(times, period_min, direction)

    # Override hotspot offset
    params_copy = dict(params)
    params_copy['hs_x_offset'] = params.get('hs_x_offset', -10)
    params_copy['hs_y_offset'] = params.get('hs_y_offset', 10)

    framelist = []
    for angle, t in tqdm(zip(angles, times), total=len(times),
                          desc="Building mring+hs-not-center movie"):
        frame = make_mring_hs_frame(angle, params_copy, obs_ref)
        frame.time = t
        framelist.append(frame)
    return eh.movie.merge_im_list(framelist, interp='linear', bounds_error=True)


def make_varbeta2_movie(params: dict, obs_ref) -> 'eh.movie.Movie':
    """Create a movie with rotating EVPA pattern (variable beta2)."""
    times, Nframes = _get_frame_times(obs_ref)
    tstart = times[0]
    tlist = np.linspace(0, times[-1] - tstart, Nframes)

    framelist = []
    for t_elapsed, t in tqdm(zip(tlist, times), total=len(times),
                              desc="Building mring-varbeta2 movie"):
        frame = make_varbeta2_frame(t_elapsed, params, obs_ref)
        frame.time = t
        framelist.append(frame)
    return eh.movie.merge_im_list(framelist, interp='linear', bounds_error=True)


# ═══════════════════════════════════════════════════════════════════════
#  GRMHD Loader & Composition
# ═══════════════════════════════════════════════════════════════════════

def load_grmhd(filepath: str, params: dict) -> 'eh.movie.Movie':
    """Load a GRMHD simulation HDF5 and optionally rotate PA.

    Args:
        filepath: path to HDF5 movie file.
        params: dict with optional 'pa_rotate_deg', 'target_flux'.

    Returns:
        ehtim.Movie
    """
    mov = eh.movie.load_hdf5(filepath)
    mov.reset_interp(bounds_error=False)

    pa = params.get('pa_rotate_deg', 0)
    if pa != 0:
        im_list = []
        for im in tqdm(mov.im_list(), desc=f"Rotating GRMHD by {pa}°"):
            im = im.rotate(np.deg2rad(pa))
            im_list.append(im)
        mov = eh.movie.merge_im_list(im_list)
        mov.reset_interp(bounds_error=False)

    target_flux = params.get('target_flux', None)
    if target_flux is not None:
        avg_flux = mov.avg_frame().total_flux()
        if avg_flux > 0:
            scale = target_flux / avg_flux
            for im in mov.im_list():
                im.imvec *= scale

    return mov


def compose_grmhd_hotspot(grmhd_mov, params: dict,
                           obs_ref=None) -> 'eh.movie.Movie':
    """Add an orbiting Gaussian hotspot to a GRMHD movie.

    Args:
        grmhd_mov: base GRMHD movie.
        params: dict with hs_flux, hs_start_angle (deg), direction, period_min.
        obs_ref: unused; kept for API consistency with other movie builders.

    Returns:
        Composed ehtim.Movie.
    """
    times = grmhd_mov.times
    hs_flux = params.get('hs_flux', 0.3)
    flux_subtract = params.get('flux_subtract', 0.3)
    period_min = params.get('period_min', 80)
    direction = params.get('direction', 'CW')
    start_angle = np.deg2rad(params.get('hs_start_angle', 0))

    period_hr = period_min / 60.0
    sign = 1 if direction == 'CW' else -1

    fov = params['fov_uas'] * eh.RADPERUAS
    npix = params['npix']
    R = params.get('ring_radius_uas', 26) * eh.RADPERUAS
    hs_fwhm = params.get('hs_fwhm_uas', 20) * eh.RADPERUAS

    combined_list = []
    grmhd_imlist = grmhd_mov.im_list()

    for i, im in tqdm(enumerate(grmhd_imlist), total=len(grmhd_imlist),
                       desc="Composing GRMHD+hs"):
        t = im.time
        t_elapsed = t - times[0]
        angle = start_angle + sign * 2 * np.pi * t_elapsed / period_hr

        # Regrid if needed
        im = im.regrid_image(fov, npix)
        combined = im.copy()

        # Subtract flux for hotspot budget
        total_f = combined.total_flux()
        if total_f > 0:
            combined.ivec *= (total_f - flux_subtract) / total_f

        # Add hotspot
        xg = R * np.cos(angle)
        yg = R * np.sin(angle)
        gauss_par = {
            'flux': hs_flux,
            'beamparams': [hs_fwhm, hs_fwhm, 0, xg, yg],
        }
        combined = combined.add_gauss(**gauss_par, pol=None)
        combined_list.append(combined)

    return eh.movie.merge_im_list(combined_list, interp='linear',
                                   bounds_error=True)


# ═══════════════════════════════════════════════════════════════════════
#  Movie Utilities
# ═══════════════════════════════════════════════════════════════════════

def extract_movie_window(mov, tstart, tstop, tshift, obs_ref,
                          pangle=0) -> 'eh.movie.Movie':
    """Extract a time window from a movie and align with observation metadata.

    Args:
        mov: source ehtim.Movie.
        tstart, tstop: UT hours for the extraction window.
        tshift: time offset applied to source movie times.
        obs_ref: reference observation (provides ra, dec, rf, mjd).
        pangle: position angle rotation (degrees), default 0.

    Returns:
        New ehtim.Movie with times in [tstart, tstop].
    """
    dtime = np.mean(np.diff(mov.times))
    nt = int((tstop - tstart) // dtime)
    times_src = np.linspace(tshift + tstart, tshift + tstop, nt)
    times_new = np.linspace(tstart, tstop, nt)

    im_list = []
    for t_src, t_new in tqdm(zip(times_src, times_new), total=nt,
                              desc="Extracting window"):
        im = mov.get_image(t_src)
        im.time = t_new
        im.mjd = obs_ref.mjd
        if pangle != 0:
            im = im.rotate(np.deg2rad(pangle))
        im_list.append(im)

    mov_new = eh.movie.merge_im_list(im_list)
    mov_new.ra = obs_ref.ra
    mov_new.dec = obs_ref.dec
    mov_new.rf = obs_ref.rf
    return mov_new


def rescale_movie_flux(mov, target_flux: float):
    """Rescale all movie frames so average total flux = target_flux."""
    avg_flux = mov.avg_frame().total_flux()
    if avg_flux <= 0:
        return mov
    scale = target_flux / avg_flux
    for im in mov.im_list():
        im.imvec *= scale
    mov.reset_interp(bounds_error=False)
    return mov


# ═══════════════════════════════════════════════════════════════════════
#  Saving & Loading with Provenance
# ═══════════════════════════════════════════════════════════════════════

def save_model(mov, filepath: str, params: dict, obs_ref=None, no_plots=False, fps=10, skip_visvar=True) -> None:
    """Save movie HDF5 + sidecar YAML with all generation parameters."""
    outdir = os.path.dirname(filepath)
    os.makedirs(outdir, exist_ok=True)
    
    # Save HDF5
    mov.save_hdf5(filepath)

    # Save FITS avg frame
    fits_path = filepath.replace('.hdf5', '.fits')
    movie_avg = mov.avg_frame();
    movie_avg.save_fits(fits_path)
    
    # Export static plot
    try:
        basename = os.path.basename(filepath).replace('.hdf5', '')
        avg_dir = os.path.join(outdir, 'avg')
        os.makedirs(avg_dir, exist_ok=True)
        png_path = os.path.join(avg_dir, f"{basename}.png")
        movie_avg.display(export_pdf=png_path, show=False);
    except Exception as e:
        print(f"  [WARN] Could not generate average plot: {e}")

    # Generate high-quality GIFs using visualize logic if obs is provided
    if obs_ref is not None and not no_plots:
        print("  Generating GIFs ...")
        try:
            from src.visualize import (process_obs_local, process_movie_parallel, compute_static_dynamic, 
                                     generate_gif_parallel, render_total_frame, render_lp_frame, plot_variance)
            import ehtim as eh
            
            obs, times = process_obs_local(obs_ref, filepath)
            if times is not None:
                npix = 160
                fov = 160 * eh.RADPERUAS
                mov.reset_interp(bounds_error=False)
                times = np.arange(times[0], times[-1]+1.0/60.0, 1.0/60.0)
                recon_frames = process_movie_parallel(mov, times, fov, npix, 16)
                recon_frames = [f for f in recon_frames if f is not None]
                recon_processed, recon_static = compute_static_dynamic(recon_frames)

                gifs_dir = os.path.join(os.path.dirname(filepath), 'gifs')
                os.makedirs(gifs_dir, exist_ok=True)
                plot_prefix = os.path.join(gifs_dir, basename)

                print(f"  -> Rendering Total Intensity GIF ({fps} FPS)")
                generate_gif_parallel(render_total_frame, recon_processed, None, times, plot_prefix, fps, 160, 16, "total", label='Model')
                print(f"  -> Rendering Linear Polarization GIF ({fps} FPS)")
                generate_gif_parallel(render_lp_frame, recon_processed, None, times, plot_prefix, fps, 160, 16, "lp", label='Model')
                if not skip_visvar:
                    print("  -> Rendering Visibility Variance")
                    plot_variance(recon_processed, None, obs, plot_prefix, fov, npix, 16, label='Model')
        except Exception as e:
            print(f"  [WARN] visualize logic failed for {basename}: {e}")

    # Clean up parameters: omit None values
    clean_params = {k: v for k, v in params.items() if v is not None}

    # Save parameter sidecar
    yaml_path = filepath.replace('.hdf5', '_params.yaml')
    params_serializable = _make_serializable(clean_params)
    with open(yaml_path, 'w') as f:
        yaml.dump(params_serializable, f, default_flow_style=False)


def load_model(filepath: str):
    """Load movie HDF5 + read sidecar YAML params.

    Returns:
        Tuple of (ehtim.Movie, dict).
    """
    mov = eh.movie.load_hdf5(filepath)
    mov.reset_interp(bounds_error=False)

    yaml_path = filepath.replace('.hdf5', '_params.yaml')
    params = {}
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            params = yaml.safe_load(f) or {}

    return mov, params


def _make_serializable(obj):
    """Convert numpy types to Python natives for YAML serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, complex):
        return {'real': float(obj.real), 'imag': float(obj.imag)}
    elif isinstance(obj, np.complexfloating):
        return {'real': float(obj.real), 'imag': float(obj.imag)}
    return obj
