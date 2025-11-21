# coding: utf-8
from .optionbase import (
    BoolOption,
    NumericOption,
    StringOption,
    StringOptionJoined,
    IntegerOption,
    TupleOption,
    FileOption
)
from .accelerad import AcceleradOptions
import warnings


class RpictOptions(AcceleradOptions):
    """rpict command options.

    Also see: https://floyd.lbl.gov/radiance/man_html/rpict.1.html
    """

    __slots__ = (
        "_vt",
        "_vp",
        "_vd",
        "_vu",
        "_vh",
        "_vv",
        "_vo",
        "_va",
        "_vs",
        "_vl",
        "_x",
        "_y",
        "_pa",
        "_pj",
        "_pm",
        "_pd",
        "_ps",
        "_pt",
        "_t",
        "_w",
        "_i",
        "_u",
        "_bv",
        "_dt",
        "_dc",
        "_dj",
        "_ds",
        "_dr",
        "_dp",
        "_dv",
        "_ss",
        "_st",
        "_av",
        "_aw",
        "_ab",
        "_aa",
        "_ar",
        "_ad",
        "_as",
        "_af",
        "_ae",
        "_ai",
        "_aE",
        "_aI",
        "_me",
        "_ma",
        "_mg",
        "_ms",
        "_lr",
        "_lw",
        "_am",
    )

    def __init__(self):
        """rpict command options."""
        AcceleradOptions.__init__(self)
        self._vt = StringOptionJoined(
            "vt",
            "view type - default: vtv",
            valid_values=["v", "l", "c", "h", "a", "s"],
            whole=False
        )
        self._vp = TupleOption(
            "vp", "view point - default: 0.000000 0.000000 0.000000", None, 3, float
        )
        self._vd = TupleOption(
            "vd", "view direction - default: 0.000000 1.000000 0.000000", None, 3, float
        )
        self._vu = TupleOption(
            "vu", "view up - default: 0.000000 0.000000 1.000000", None, 3, float
        )
        self._vh = NumericOption("vh", "view horizontal size - default: 45.000000")
        self._vv = NumericOption("vv", "view vertical size - default: 45.000000")
        self._vo = NumericOption("vo", "view fore clipping plane - default: 0.000000")
        self._va = NumericOption("va", "view aft clipping plane - default: 0.000000")
        self._vs = NumericOption("vs", "view shift - default: 0.000000")
        self._vl = NumericOption("vl", "view lift - default: 0.000000")
        self._x = IntegerOption("x", "x resolution - default: 512")
        self._y = IntegerOption("y", "y resolution - default: 512")
        self._pa = NumericOption("pa", "pixel aspect ratio - default: 1.000000")
        self._pj = NumericOption("pj", "pixel jitter - default: 0.670000")
        self._pm = NumericOption("pm", "pixel motion - default: 0.000000")
        self._pd = NumericOption("pd", "pixel depth-of-field - default: 0.000000")
        self._ps = IntegerOption("ps", "pixel sample - default: 4")
        self._pt = NumericOption("pt", "pixel threshold - default: 0.050000")
        self._t = IntegerOption("t", "time between reports - default: 0")
        self._w = BoolOption("w", "warning messages - default: True")
        self._i = BoolOption("i", "irradiance calculation - default: False")
        self._u = BoolOption(
            "u", "correlated quasi-Monte Carlo sampling - default: False"
        )
        self._bv = BoolOption("bv", "back face visibility - default: True")
        self._dt = NumericOption("dt", "direct threshold - default: 0.050000")
        self._dc = NumericOption("dc", "direct certainty - default: 0.500000")
        self._dj = NumericOption("dj", "direct jitter - default: 0.000000")
        self._ds = NumericOption("ds", "direct sampling - default: 0.250000")
        self._dr = IntegerOption("dr", "direct relays - default: 1")
        self._dp = IntegerOption("dp", "direct pretest density - default: 512")
        self._dv = BoolOption("dv", "direct visibility - default: True")
        self._ss = NumericOption("ss", "specular sampling - default: 1.000000")
        self._st = NumericOption("st", "specular threshold - default: 0.150000")
        self._av = TupleOption(
            "av", "ambient value - default: 0.000000 0.000000 0.000000", None, 3, float
        )
        self._aw = IntegerOption("aw", "ambient value weight - default: 0")
        self._ab = IntegerOption("ab", "ambient bounces - default: 0")
        self._aa = NumericOption("aa", "ambient accuracy - default: 0.200000")
        self._ar = IntegerOption("ar", "ambient resolution - default: 64")
        self._ad = IntegerOption("ad", "ambient divisions - default: 512")
        self._as = IntegerOption("as_", "ambient super-samples - default: 128")
        self._af = FileOption('af', 'ambient cache file (.amb)')
        self._ae = StringOption('ae', 'ambient excluded modifier')
        self._ai = StringOption('ai', 'ambient included modifier')
        self._aE = FileOption('aE', 'ambient excluded modifiers file')
        self._aI = FileOption('aI', 'ambient included modifiers file')
        self._me = TupleOption(
            "me",
            "mist extinction coefficient - default: 0.00e+00 0.00e+00 0.00e+00",
            None,
            3,
            float,
        )
        self._ma = TupleOption(
            "ma",
            "mist scattering albedo - default: 0.000000 0.000000 0.000000",
            None,
            3,
            float,
        )
        self._mg = NumericOption(
            "mg", "mist scattering eccentricity - default: 0.000000"
        )
        self._ms = NumericOption("ms", "mist sampling distance - default: 0.000000")
        self._lr = IntegerOption("lr", "limit reflection - default: 7")
        self._lw = NumericOption("lw", "limit weight - default: 1.00e-03")
        self._am = NumericOption("am", "max photon search radius - default: 0.0")
        self._on_setattr_check = True

    def _on_setattr(self):
        """This method executes after setting each new attribute.

        Use this method to add checks that are necessary for OptionCollection. For
        instance in rpict option collection -dj and -ps don't go together very well.
        You can include a check to ensure this is always correct.
        """

        if self._ai.is_set and self._ae.is_set:
            raise ValueError(
                'Both ai and ae are set. The program can use either an include list or'
                ' an exclude list, but not both.'
            )

        if self._aI.is_set and self._aE.is_set:
            raise ValueError(
                'Both aI and aE are set. The program can use either an include list or'
                ' an exclude list, but not both.'
            )

        if self._dj.is_set and self._ps.is_set:
            if not (self._dj > 0.0 and self._ps != -1):
                warnings.warn(
                    'It is usually wise to turn off image sampling when using direct'
                    ' jitter.'
                )
        if self._i.value and self._dv.value:
            warnings.warn(
                'If irradiance values are requested, it is better to keep -dv off'
                ' so that light sources do not appear with their original radiance'
                ' values.'
            )

    @property
    def vt(self):
        """view type perspective - default: vtv

        *   'v' sets a perspective view.
        *   'l' sets parallel view.
        *   'c' sets a cylindrical panaroma. This view is like a standard perspective
            vertically, but projected on a cylinder horizontally, like a soupcan's
            eye-view.
        *   'h' sets a hemispherical fisheye view. This is a projection of the hemisphere
            onto a circle. The maximum view angle for this type is 180 degrees.
        *   'a' sets an angular fisheye view. An angular fisheye view is defined such
            that distance from the center of the image is proportional to the angle
            from the central view direction. An angular fisheye can display a full 360
            degrees.
        *   's' sets a planisphere (stereographic) view. A planisphere fisheye view
            maintains angular relationships between lines, and is commonly used for
            sun path analysis. This is more commonly known as a stereographic projection.
        """
        return self._vt

    @vt.setter
    def vt(self, value):
        self._vt.value = value

    @property
    def vp(self):
        """view point - default: 0.000000 0.000000 0.000000

        Set the view point to x y z . This is the focal point of a perspective
        view or the center of a parallel projection.
        """
        return self._vp

    @vp.setter
    def vp(self, value):
        self._vp.value = value

    @property
    def vd(self):
        """view direction - default: 0.000000 1.000000 0.000000

        Set the view direction vector to xd yd zd . The length of this
        vector indicates the focal distance as needed by the -pd option.
        """
        return self._vd

    @vd.setter
    def vd(self, value):
        self._vd.value = value

    @property
    def vu(self):
        """view up - default: 0.000000 0.000000 1.000000

        Set the view up vector (vertical direction) to xd yd zd.
        """
        return self._vu

    @vu.setter
    def vu(self, value):
        self._vu.value = value

    @property
    def vh(self):
        """view horizontal size - default: 45.000000

        Set the view horizontal size. For a perspective projection
        (including fisheye views), this size is the horizontal field of view
        (in degrees). For a parallel projection, this size is the view width in world
        coordinates.
        """
        return self._vh

    @vh.setter
    def vh(self, value):
        self._vh.value = value

    @property
    def vv(self):
        """view vertical size - default: 45.000000

        Set the view vertical size.
        """
        return self._vv

    @vv.setter
    def vv(self, value):
        self._vv.value = value

    @property
    def vo(self):
        """view fore clipping plane - default: 0.000000

        Set the view fore clipping plane at a distance from the view point.
        The plane will be perpendicular to the view direction for perspective and
        parallel view types. For fisheye view types, the clipping plane is actually
        a clipping sphere, centered on the view point with radius "vo". objects in
        from of this imaginary surface will not be visible. This may be useful for
        seeing through walls (to get a longer perspective from an exterior view point.)
        or for incremental rendering. A value of zero implies no foreground clipping.
        A negative value produces some interesting effects, since it creates an
        inverted image for objects behind the viewport. This possibility is provided
        mostly for the purpose of rendering stereographic holograms.
        """
        return self._vo

    @vo.setter
    def vo(self, value):
        self._vo.value = value

    @property
    def va(self):
        """view aft clipping plane - default: 0.000000

        Set the view aft clipping plane at a distance of "va" from the view
        point. Like the view fore plane, it will be perpendicular to the view direction
        for perspective and parallel view types. For fisheye view types, the clipping
        plane is actually a clipping sphere, centered on the view point with radius "va".
        Objects behind this imaginary surface will not be visible. A value of zero
        means no aft clipping, and is the only way to see infinitely distant objects
        such as the sky.
        """
        return self._va

    @va.setter
    def va(self, value):
        self._va.value = value

    @property
    def vs(self):
        """view shift - default: 0.000000

        Set value to shift the view. This is the amount the actual image will be shifted
        to the right of the specified view. This is option is useful for generating
        skewed perspectives or rendering an image a piece at a time. A value of 1 means
        that the rendered image starts just to the right of the normal view. A value of
        −1 would be to the left. Larger or fractional values are permitted as well.
        """
        return self._vs

    @vs.setter
    def vs(self, value):
        self._vs.value = value

    @property
    def vl(self):
        """view lift - default: 0.000000

        Set the view lift. This is the amount the actual image will be
        lifted up from the specified view, similar to the -vs option.
        """
        return self._vl

    @vl.setter
    def vl(self, value):
        self._vl.value = value

    @property
    def x(self):
        """x resolution - default: 512

        Set the maximum x resolution.
        """
        return self._x

    @x.setter
    def x(self, value):
        self._x.value = value

    @property
    def y(self):
        """y resolution - default: 512

        Set the maximum y resolution.
        """
        return self._y

    @y.setter
    def y(self, value):
        self._y.value = value

    @property
    def pa(self):
        """pixel aspect ratio - default: 1.000000

        Set the pixel aspect ratio (height over width). Either the x or the y
        resolution will be reduced so that the pixels have this ratio for the specified
        view. If this value is zero, then the x and y resolutions will adhere to the
        given maxima.
        """
        return self._pa

    @pa.setter
    def pa(self, value):
        self._pa.value = value

    @property
    def pj(self):
        """pixel jitter - default: 0.670000

        Set the pixel sample jitter to frac. Distributed ray-tracing performs
        anti-aliasing by randomly sampling over pixels. A value of one will randomly
        distribute samples over full pixels. A value of zero samples pixel centers only.
        A value between zero and one is usually best for low-resolution images.
        """
        return self._pj

    @pj.setter
    def pj(self, value):
        self._pj.value = value

    @property
    def pm(self):
        """pixel motion - default: 0.000000

        Set the pixel motion blur to frac. In an animated sequence, the
        exact view will be blurred between the previous view and the next view as
        though a shutter were open this fraction of a frame time. (See the −S option
        regarding animated sequences.) The first view will be blurred according to
        the difference between the initial view set on the command line and the first
        view taken from the standard input. It is not advisable to use this option
        in combination with the pmblur(1) program, since one takes the place of
        the other. However, it may improve results with pmblur to use a very small
        fraction with the −pm option, to avoid the ghosting effect of too few
        time samples.
        """
        return self._pm

    @pm.setter
    def pm(self, value):
        self._pm.value = value

    @property
    def pd(self):
        """pixel depth-of-field - default: 0.000000

        Set the pixel depth-of-field aperture to a diameter of
        dia (in world coordinates). This will be used in conjunction with the
        view focal distance, indicated by the length of the view direction
        vector given in the −vd option. It is not advisable to use this option
        in combination with the pdfblur(1) program, since one takes the place
        of the other. However, it may improve results with pdfblur to use a
        very small fraction with the −pd option, to avoid the ghosting effect
        of too few samples.
        """
        return self._pd

    @pd.setter
    def pd(self, value):
        self._pd.value = value

    @property
    def ps(self):
        """pixel sample - default: 4

        Set the pixel sample spacing to the integer size.
        This specifies the sample spacing (in pixels) for adaptive
        subdivision on the image plane.
        """
        return self._ps

    @ps.setter
    def ps(self, value):
        self._ps.value = value

    @property
    def pt(self):
        """pixel threshold - default: 0.050000

        Set the pixel sample tolerance to frac. If two samples differ
        by more than this amount, a third sample is taken between them.
        """
        return self._pt

    @pt.setter
    def pt(self, value):
        self._pt.value = value

    @property
    def t(self):
        """time between reports - default: 0

        Set the time between progress reports to sec.
        A progress report writes the number of rays traced,
        the percentage completed, and the CPU usage to the standard error.
        Reports are given either automatically after the specified interval,
        or when the process receives a continue (−CONT) signal (see kill(1)).
        A value of zero turns automatic reporting off.
        """
        return self._t

    @t.setter
    def t(self, value):
        self._t.value = value

    @property
    def w(self):
        """warning messages - default: True

        Boolean switch for warning messages. The default is to print warnings,
        so the first appearance of this option turns them off.
        """
        return self._w

    @w.setter
    def w(self, value):
        self._w.value = value

    @property
    def i(self):
        """irradiance calculation - default: False

        Boolean switch to compute irradiance rather than radiance values.
        This only affects the final result, substituting a Lambertian surface
        and multiplying the radiance by pi. Glass and other transparent surfaces
        are ignored during this stage. Light sources still appear with their
        original radiance values, though the −dv option (above) may be used
        to override this.
        """
        return self._i

    @i.setter
    def i(self, value):
        self._i.value = value

    @property
    def u(self):
        """correlated quasi-Monte Carlo sampling - default: False

        Boolean switch to control uncorrelated random sampling. When "off",
        a low-discrepancy sequence is used, which reduces variance but can result
        in a brushed appearance in specular highlights. When "on",
        pure Monte Carlo sampling is used in all calculations.
        """
        return self._u

    @u.setter
    def u(self, value):
        self._u.value = value

    @property
    def bv(self):
        """back face visibility - default: True

        Boolean switch for back face visibility. With this switch off,
        back faces of opaque objects will be invisible to all rays.
        This is dangerous unless the model was constructed such that all
        surface normals on opaque objects face outward. Although turning
        off back face visibility does not save much computation time under
        most circumstances, it may be useful as a tool for scene debugging,
        or for seeing through one-sided walls from the outside. This option
        has no effect on transparent or translucent materials.
        """
        return self._bv

    @bv.setter
    def bv(self, value):
        self._bv.value = value

    @property
    def dt(self):
        """direct threshold - default: 0.050000

        Set the direct threshold to frac. Shadow testing will stop when
        the potential contribution of at least the next and at most all remaining
        light source samples is less than this fraction of the accumulated value.
        (See the −dc option) The remaining light source contributions are
        approximated statistically. A value of zero means that all light source
        samples will be tested for shadow.
        """
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt.value = value

    @property
    def dc(self):
        """direct certainty - default: 0.500000

        Set the direct certainty to frac. A value of one guarantees
        that the absolute accuracy of the direct calculation will be equal to
        or better than that given in the −dt specification. A value of zero only
        insures that all shadow lines resulting in a contrast change greater than
        the −dt specification will be calculated.
        """
        return self._dc

    @dc.setter
    def dc(self, value):
        self._dc.value = value

    @property
    def dj(self):
        """direct jitter - default: 0.000000

        Set the direct jittering to frac. A value of zero samples each
        source at specific sample points (see the −ds option), giving a
        smoother but somewhat less accurate rendering. A positive value causes
        rays to be distributed over each source sample according to its size,
        resulting in more accurate penumbras. This option should never be greater
        than 1, and may even cause problems (such as speckle) when the value is
        smaller. A warning about aiming failure will issued if frac is too large.
        It is usually wise to turn off image sampling when using direct jitter by
        setting −ps to 1.
        """
        return self._dj

    @dj.setter
    def dj(self, value):
        self._dj.value = value

    @property
    def ds(self):
        """direct sampling - default: 0.250000

        Set the direct sampling ratio to frac. A light source will be
        subdivided until the width of each sample area divided by the distance
        to the illuminated point is below this ratio. This assures accuracy in
        regions close to large area sources at a slight computational expense.
        A value of zero turns source subdivision off, sending at most one shadow
        ray to each light source.
        """
        return self._ds

    @ds.setter
    def ds(self, value):
        self._ds.value = value

    @property
    def dr(self):
        """direct relays - default: 1

        Set the number of relays for secondary sources to N. A value of 0 means
        that secondary sources will be ignored. A value of 1 means that sources will
        be made into first generation secondary sources; a value of 2 means that
        first generation secondary sources will also be made into second generation
        secondary sources, and so on.
        """
        return self._dr

    @dr.setter
    def dr(self, value):
        self._dr.value = value

    @property
    def dp(self):
        """direct pretest density - default: 512

        Set the secondary source presampling density to D. This is the number of
        samples per steradian that will be used to determine ahead of time whether
        or not it is worth following shadow rays through all the reflections and/or
        transmissions associated with a secondary source path. A value of 0 means
        that the full secondary source path will always be tested for shadows if
        it is tested at all.
        """
        return self._dp

    @dp.setter
    def dp(self, value):
        self._dp.value = value

    @property
    def dv(self):
        """direct visibility - default: True

        Boolean switch for light source visibility. With this switch off,
        sources will be black when viewed directly although they will still
        participate in the direct calculation. This option may be desirable
        in conjunction with the −i option so that light sources do not appear
        in the output.
        """
        return self._dv

    @dv.setter
    def dv(self, value):
        self._dv.value = value

    @property
    def ss(self):
        """specular sampling - default: 1.000000

        Set the specular sampling to samp. For values less than 1,
        this is the degree to which the highlights are sampled for rough specular
        materials. A value greater than one causes multiple ray samples to be sent
        to reduce noise at a commmesurate cost. A value of zero means that no
        jittering will take place, and all reflections will appear sharp even
        when they should be diffuse. This may be desirable when used in combination
        with image sampling (see −ps option above) to obtain faster renderings.
        """
        return self._ss

    @ss.setter
    def ss(self, value):
        self._ss.value = value

    @property
    def st(self):
        """specular threshold - default: 0.150000

        Set the specular sampling threshold to frac.
        This is the minimum fraction of reflection or transmission,
        under which no specular sampling is performed. A value of zero
        means that highlights will always be sampled by tracing reflected
        or transmitted rays. A value of one means that specular sampling
        is never used. Highlights from light sources will always be correct,
        but reflections from other surfaces will be approximated using an
        ambient value. A sampling threshold between zero and one offers a
        compromise between image accuracy and rendering time.
        """
        return self._st

    @st.setter
    def st(self, value):
        self._st.value = value

    @property
    def av(self):
        """ambient value - default: 0.000000 0.000000 0.000000

        Set the ambient value to a radiance of red grn blu. This is the final
        value used in place of an indirect light calculation. If the number of
        ambient bounces is one or greater and the ambient value weight is non-zero
        (see -aw and -ab), this value may be modified by the computed
        indirect values to improve overall accuracy.
        """
        return self._av

    @av.setter
    def av(self, value):
        self._av.value = value

    @property
    def aw(self):
        """ambient value weight - default: 0

        Set the relative weight of the ambient value given with the -av option to N.
        As new indirect irradiances are computed, they will modify the default ambient
        value in a moving average, with the specified weight assigned to the initial
        value given on the command and all other weights set to 1. If a value of 0 is
        given with this option, then the initial ambient value is never modified.
        This is the safest value for scenes with large differences in indirect
        contributions, such as when both indoor and outdoor (daylight) areas
        are visible.
        """
        return self._aw

    @aw.setter
    def aw(self, value):
        self._aw.value = value

    @property
    def ab(self):
        """ambient bounces - default: 0

        Set the number of ambient bounces to N. This is the maximum number
        of diffuse bounces computed by the indirect calculation. A value of
        zero implies no indirect calculation.
        """
        return self._ab

    @ab.setter
    def ab(self, value):
        self._ab.value = value

    @property
    def aa(self):
        """ambient accuracy - default: 0.200000

        Set the ambient accuracy. This value will approximately equal the
        error from indirect illuminance interpolation. A value of zero implies
        no interpolation.
        """
        return self._aa

    @aa.setter
    def aa(self, value):
        self._aa.value = value

    @property
    def ar(self):
        """ambient resolution - default: 64

        Set the ambient resolution. This number will determine the maximum density
        of ambient values used in interpolation. Error will start to increase
        on surfaces spaced closer than the scene size divided by the ambient
        resolution. The maximum ambient value density is the scene size times
        the ambient accuracy (see the −aa option) divided by the ambient
        resolution. The scene size can be determined using getinfo(1) with the
        −d option on the input octree. A value of zero is interpreted as
        unlimited resolution.
        """
        return self._ar

    @ar.setter
    def ar(self, value):
        self._ar.value = value

    @property
    def ad(self):
        """ambient divisions - default: 512

        Set the number of ambient divisions to N. The error in the Monte Carlo
        calculation of indirect illuminance will be inversely proportional to the
        square root of this number. A value of zero implies no indirect calculation.
        """
        return self._ad

    @ad.setter
    def ad(self, value):
        self._ad.value = value

    @property
    def as_(self):
        """ambient super-samples - default: 128

        Set the number of ambient super-samples to N. Super-samples are applied
        only to the ambient divisions which show a significant change.
        """
        return self._as

    @as_.setter
    def as_(self, value):
        self._as.value = value

    @property
    def af(self):
        """ambient cache file (.amb)

        This is where indirect illuminance will be stored and retrieved. Normally,
        indirect illuminance values are kept in memory and lost when the program
        finishes or dies. By using a file, different invocations can share
        illuminance values, saving time in the computation. Also, by creating an
        ambient file during a low resolution rendering, better results can be
        obtained in a second high resolution pass. The ambient file is in a
        machine-independent binary format which may be examined with lookamb.

        The ambient file may also be used as a means of communication and data
        sharing between simultaneously executing processes. The same file may be
        used by multiple processes, possibly running on different machines and
        accessing the file via the network (ie. nfs). The network lock manager
        lockd is used to insure that this information is used consistently.

        If any calculation parameters are changed or the scene is modified, the
        old ambient file should be removed so that the calculation can start over
        from scratch. For convenience, the original ambient parameters are listed
        in the header of the ambient file. Getinfo(1) may be used to print out
        this information.
        """
        return self._af

    @af.setter
    def af(self, value):
        self._af.value = value

    @property
    def ae(self):
        """ambient excluded modifier.

        Append mod to the ambient exclude list, so that it will not be considered
        during the indirect calculation. This is a hack for speeding the indirect
        computation by ignoring certain objects. Any object having mod as its
        modifier will get the default ambient level rather than a calculated value.
        Any number of excluded modifiers may be given, but each must appear in a
        separate option.
        """
        return self._ae

    @ae.setter
    def ae(self, value):
        self._ae.value = value

    @property
    def ai(self):
        """ambient included modifier.

        Add mod to the ambient include list, so that it will be considered during
        the indirect calculation. The program can use either an include list or
        an exclude list, but not both.
        """
        return self._ai

    @ai.setter
    def ai(self, value):
        self._ai.value = value

    @property
    def aE(self):
        """ambient excluded modifiers file

        Same as −ae, except read modifiers to be excluded from file.
        The RAYPATH environment variable determines which directories are searched
        for this file. The modifier names are separated by white space in the file.
        """
        return self._aE

    @aE.setter
    def aE(self, value):
        self._aE.value = value

    @property
    def aI(self):
        """ambient included modifiers file

        Same as −ai, except read modifiers to be included from file.
        """
        return self._aI

    @aI.setter
    def aI(self, value):
        self._aI.value = value

    @property
    def me(self):
        """mist extinction coefficient - default: 0.00e+00 0.00e+00 0.00e+00

        Set the global medium extinction coefficient to the indicated color,
        in units of 1/distance (distance in world coordinates). Light will be
        scattered or absorbed over distance according to this value. The ratio
        of scattering to total scattering plus absorption is set by -ma option.
        """
        return self._me

    @me.setter
    def me(self, value):
        self._me.value = value

    @property
    def ma(self):
        """mist scattering albedo - default: 0.000000 0.000000 0.000000

        Set the global medium albedo to the given value between 0 0 0 and 1 1 1.
        A zero value means that all light not transmitted by the medium is absorbed.
        A unitary value means that all light not transmitted by the medium is
        scattered in some new direction. The isotropy of scattering is determined
        by the -mg option.
        """
        return self._ma

    @ma.setter
    def ma(self, value):
        self._ma.value = value

    @property
    def mg(self):
        """mist scattering eccentricity - default: 0.000000

        Set the medium Heyney-Greenstein eccentricity parameter.
        This parameter determines how strongly scattering favors the forward
        direction. A value of 0 indicates perfectly isotropic scattering. As
        this parameter approaches 1, scattering tends to prefer the forward direction.
        """
        return self._mg

    @mg.setter
    def mg(self, value):
        self._mg.value = value

    @property
    def ms(self):
        """mist sampling distance - default: 0.000000

        Set the medium sampling distance, in world coordinate units.
        During source scattering, this will be the average distance between
        adjacent samples. A value of 0 means that only one sample will be taken
        per light source within a given scattering volume.
        """
        return self._ms

    @ms.setter
    def ms(self, value):
        self._ms.value = value

    @property
    def lr(self):
        """limit reflection - default: 7

        Limit reflections to a maximum of this value, if the value is a positive integer.
        If the value is zero, then Russian roulette is used for ray termination,
        and the -lw setting must be positive. If the value is a negative
        integer, then this sets the upper limit of reflections past which
        Russian roulette will be used. In scenes with dielectrics and total
        internal reflection, a setting of 0 (no limit) may cause a stack overflow.
        """
        return self._lr

    @lr.setter
    def lr(self, value):
        self._lr.value = value

    @property
    def lw(self):
        """limit weight - default: 1.00e-03

        Limit the weight of each ray to a minimum of frac. During ray-tracing, a
        record is kept of the estimated contribution (weight) a ray would have in
        the image. If this weight is less than the specified minimum and the -lr
        setting is positive, the ray is not traced. Otherwise, Russian
        roulette is used to continue rays with a probability equal to the ray
        weight divided by the given frac.
        """
        return self._lw

    @lw.setter
    def lw(self, value):
        self._lw.value = value

    @property
    def am(self):
        """max photon search radius - default: 0.0"""
        return self._am

    @am.setter
    def am(self, value):
        self._am.value = value
