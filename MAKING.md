## Casting procedure

### Supplies

* PVA filament for printing the mold. I've been using 3dxtech PVA.

* Glue stick to help with adhesion. I use a PEI build plate, but still
  find that glue stick is helpful for printing PVA.

* Thermometer to measure temperature of the molten metal, capable of
  measuing up to 200-220 °C. A basic digital meat thermometer does the
  trick.

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

* Hot plate and pot for melting the alloy. If you use your stove and
  favorite saucepan I'm not responsible for damage. I use a small 4"
  500 W hot plate and a similarly sized heavy stainless steel butter
  melting pot.

* Safety equipment: goggles, oven mitts, long sleeved shirt and pants,
  shoes, tongs. The molten metal can burn you badly if you mishandle
  it, and you really don't want any splashing into your eyes.

* Container ideally a little wider than the printed mold and taller
  than the piece itself, but a little shorter than the top of the
  funnel, making pouring the molten metal into the mold easier.

* Cool water to pour over the mold to cool it and keep it from
  deforming until the hot metal has started to solidify.

* Optional: heated magnetric stirrer.

* Files and sand paper to finish the model, and gloves to keep the
  finely divided metal powder from staining your fingers.

* Power supply and electroplating solutions for copper and/or nickel
  plating the piece if desired.


### Slicing

* Load the .3mf file into the slicer, and when the dialog asks to
  treat this as a single multi-part object, say yes.

* Even though we have multiple parts, this is a single-filament print,
  so set both parts to use the same extruder, and disable wipe tower
  if necessary (Print Settings, Multiple Extruders, Disable).

* PrusaSlicer doesn't have a generic PVA profile, so here's what I
  do. I have this saved as a "PVA for molds" profile.

    * Start with "PrimaSelect PVA+" profile.

    * Under cooling, select keep fan always on. This helps avoid
      curling on sharp overhangs.

    * Bump temperature by 10 °C to 210 °C for all layers. This helps
      with adhesion and compenstates for the fan.

* I've successfully used the following layer height settings:

    * for 0.3 mm nozzle: 0.12 mm STRUCTURAL

    * for 0.6 mm nozzle: 0.15 mm STRUCTURAL

* Print Settings, Layers and Perimeters (I have this saved as a "for
  molds" print profile)

    * Solid layers, top: 0
    * Solid layers, bottom: 0
    * Perimeters: 1000
    * Infill: 25% rectilinear

  The mold itself will print solid, all perimeters. The 25%
  rectilinear infill setting is a good choice for the support, but we
  also need to modify the support perimeters as follows:

* Right click on the support, and select "Layers and perimeters" and
  change:

    * Perimeters: 0

  Then the support structure will print as a rectilinear infill
  "sponge", no perimeters. This allows the cooling water to reach the
  mold.

* For some of the models that I've uploaded the slicer may warn you
  of inadequate bed adhesion, probably in part because the infill forming
  the support is being printed directly on the plate without a solid first
  layer. However I've found that with the 25% rectilinear infill and glue
  stick on PEI the bed adhesion is more than sufficient, and in fact it
  can be a little hard to remove the PVA.

### Printing the mold

I use a PEI build plate, but still have problems with adhesion. I find
that a thin layer of glue stick solves the problem (in fact maybe a
bit too well as it can be little hard to get the PVA filament off the
build plate).

### Pouring

* Melt the alloy, heating to 190-200 °C. This is well above its melting
  point, but we need to make sure it will fill the mold before
  solidifying. For smaller pieces you may need a bit higher
  temperature so it fills the mold, and for larger pieces you may want
  a bit lower temperature so the mold doesn't deform before the metal
  solidifies.

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
hard sugar candy. I've found two approaches that work.

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

* Sand starting with 200 grit sandpaper and work your way up to 1000
  or so, depending on your patience and the type of finish you want

* The finely divided tin and bismuth metal can stain your fingers, so
  I like to wear rubber gloves while doing this.

* The alloy takes well to copper electroplating, and you can then
  electroplate nickel over the top of the copper for a nice bright
  finish.


