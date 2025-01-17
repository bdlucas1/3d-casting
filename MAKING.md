## Casting procedure

### Supplies

* PVA filament for printing the mold; I've been using 3dxtech PVA. PVA
  is a bit finicky to print with, and it absorbs moisture easily. If
  you encounter issues like poor bed adhesion, poor layer-to-layer
  adehesion, difficulty printing the supports, or excessive
  brittleness, try drying the filament thoroughly. I've even had fresh
  rolls that required drying before they printed well.

* Low-melting eutectic tin-bismuth alloy. This melts at 138 °C (281
  °F), and it's sold as a substitute for lead for making fishing
  weights.  I buy mine from
  [RotoMetals](https://www.rotometals.com/lead-free-fishing-tackle-weight-bismuth-tin-alloy-281).
  Be sure to get the eutectic alloy identified by its (Fahrenheit)
  melting point "281".

  There are other alloys with lower melting points but they contain
  either lead (toxic, particularly if you're going to be filing and
  sanding it), cadmium (really toxic), and/or indium (insanely
  expensive).

* Hot plate, pot for melting the alloy, and a trivet or heat-resistant
  pad to set the hot pot on after pouring the metal.  I use a small 4"
  500 W hot plate and a similarly sized heavy stainless steel butter
  melting pot. (If you use your stove and favorite saucepan I'm not
  responsible for damage.)

* I put the hotplate on a timed outlet and set it for half an hour
  because I don't trust myself to always remember to turn it off.

* Thermometer to measure temperature of the molten metal, capable of
  measuing up to 200-220 °C. A basic digital meat thermometer does the
  trick.

* Safety equipment: goggles, oven mitts, long sleeved shirt and pants,
  shoes, tongs. The molten metal can burn you badly if you mishandle
  it, and you really don't want any splashing into your eyes.

* Container (plastic will do) to place the mold in when pouring. It should
  be wider than the printed mold and ideally taller than the piece itself,
  but a little shorter than the top of the funnel, making pouring
  the molten metal into the mold easier.

* Cool water to pour over the mold to cool it and keep it from
  deforming until the hot metal has started to solidify.

* Optional: heated magnetric stirrer.

* Files and sand paper or Scotchbrite pads to finish the model, and
  gloves to keep the finely divided metal powder from staining your
  fingers.

* Power supply and electroplating solutions for copper and/or nickel
  plating the piece if desired.


### Slicing

These instructions are for PrusaSlicer. Your slicer probably has
equivalent settings.

* Filament

  PrusaSlicer doesn't have a generic PVA profile so I started with
  PrimaSelect PVA+, turned fan on to help with curled overhangs, and
  bumped temp by 15 °C to compensate and to help with adhesion.

    * "PrimaSelect PVA+" profile
    * Cooling: keep fan always on
    * Temperature: bump by 15 °C to 215 °C for all layers

* Print Settings, Support material

  Most models are printed at odd angles to avoid flat top surfaces and
  therefore require support. Setting contact distances and interface
  pattern spacing to 0 and interface layers to 2 makes the support hug
  the outside of the mold; since the support is on the outside of the
  mold it doesn't affect the quality of the part. Different models
  have different ideal support overhang setting. The honeycomb support
  pattern is much sturdier for PVA (albeit a bit slower).  I found
  using a raft to be crucial for adhesion with PVA.

    * Overhang threshold: see model description
    * Raft layers: 2
    * Top contact Z distance: 0
    * Pattern: Honeycomb
    * Top interface layers: 2
    * Interface pattern spacing: 0
    * Support on build plate only: yes
    * XY separation between object and support: 0

* Print Settings, Layers and Perimeters

  The mold is made solid using perimeters. 

    * Solid layers, top: 0
    * Solid layers, bottom: 0
    * Perimeters: 1000
    * Seam position: rear

### Pouring

* Melt the alloy, heating to 180-200 °C. This is well above its
  melting point, but we need to make sure it will fill the mold before
  solidifying. For smaller pieces you may need a temperature at the
  higher end of the range so it fills the mold, and for larger pieces
  you may want a temperature at the lower end of the range so the mold
  doesn't deform before the metal solidifies.

  This of course is well above the glass transition temperature of the
  PVA, but the metal cools quickly on contact with the plastic, and
  the water bath will also help maintain the rigidity of the outer
  layers of the mold.

* Pour the molten metal into the funnel, stopping when either the
  funnel fills up and does not drain further, or the molten metal
  starts coming out of the tops of the vents.

* Immediately, before the mold has time to begin to deform, fill the
  container with the cool water, up to the top of the mold. This will
  help to maintain the rigidity of the mold while the metal
  solidifies.

* Wait a few minutes to give the metal a chance to solidify,
  particularly if it's a heavy piece, before dissolving the mold.

### Dissolving the mold

PVA is water soluble, but it requires some thermal and/or mechanical
encouragement. Dissolving a hunk of PVA is a bit like dissolving a
hard sugar candy. I've found two approaches that work:

* Place the mold in a water bath on a heated magnetic stirring
  plate set to 60-70 °C and turn the stirrer on. Come back in a
  few hours and the mold should be dissolved.

* If you don't have a heated magnetic stirrer (and aren't tempted by
  this opportunity to buy one), place the mold in a pyrex bowl, heat
  some water to boiling, and pour it over the mold, submerging
  it. When the water cools to room temperature, use your fingers or an
  abrasive scrubber to remove as much of the semi-dissolved mold as
  you can. Repeat this process a few times until the mold is
  completely removed.

###  Finishing the piece

* Snip or snap off the feed and vent structures, and file off the
  stubs.

* Sand starting with 200-400 grit sandpaper and work your way up to
  1000 or so, depending on your patience and the type of finish you
  want. I like to use sandpaper or a file for large flat areas, and
  Scotchbrite abrasive pads for curved or detailed areas.

* The finely divided tin and bismuth metal can stain your fingers, so
  I like to wear rubber gloves while doing this.

* The alloy takes well to copper electroplating, and you can then
  electroplate nickel over the top of the copper for a nice bright
  finish.


