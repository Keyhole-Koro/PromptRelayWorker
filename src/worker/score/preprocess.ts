import sharp from "sharp";

export type GrayImage = {
  width: number;
  height: number;
  data: Uint8ClampedArray;
};

export type RgbImage = {
  width: number;
  height: number;
  data: Uint8Array;
};

export type ImageVariants = {
  composition: GrayImage;
  detail: GrayImage;
  color: RgbImage;
  embeddingPng: Buffer;
};

async function toGrayRaw(pipeline: sharp.Sharp): Promise<GrayImage> {
  const { data, info } = await pipeline
    .grayscale()
    .raw()
    .toBuffer({ resolveWithObject: true });

  return {
    width: info.width,
    height: info.height,
    data: new Uint8ClampedArray(data),
  };
}

async function toRgbRaw(pipeline: sharp.Sharp): Promise<RgbImage> {
  const { data, info } = await pipeline
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  return {
    width: info.width,
    height: info.height,
    data: new Uint8Array(data),
  };
}

function baseSquare(input: Buffer): sharp.Sharp {
  return sharp(input)
    .rotate()
    .resize(512, 512, {
      fit: "contain",
      background: { r: 0, g: 0, b: 0, alpha: 1 },
    })
    .removeAlpha();
}

export async function makeImageVariants(input: Buffer): Promise<ImageVariants> {
  const base = baseSquare(input);

  const [composition, detail, color, embeddingPng] = await Promise.all([
    toGrayRaw(base.clone().blur(3).resize(64, 64, { fit: "fill" })),
    toGrayRaw(
      base
        .clone()
        .convolve({
          width: 3,
          height: 3,
          kernel: [0, -1, 0, -1, 5, -1, 0, -1, 0],
        })
        .resize(128, 128, { fit: "fill" }),
    ),
    toRgbRaw(base.clone().resize(256, 256, { fit: "fill" })),
    base.clone().png().toBuffer(),
  ]);

  return {
    composition,
    detail,
    color,
    embeddingPng,
  };
}
