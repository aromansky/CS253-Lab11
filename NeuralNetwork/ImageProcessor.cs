using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging; // Обязательно для PixelFormat
using AForge.Imaging;
using AForge.Imaging.Filters;

namespace AIMLTGBot
{
    public static class ImageProcessor
    {
        public const int InputWidth = 64;
        public const int InputHeight = 32;
        public const int InputSize = InputWidth * InputHeight;

        private const int BinaryThreshold = 100;

        public static double[] ProcessImage(Bitmap bitmap)
        {
            // Приводим любое входящее изображение к формату 24bpp RGB.
            Bitmap sourceImage;
            bool tempSource = false;

            if (bitmap.PixelFormat != PixelFormat.Format24bppRgb)
            {
                sourceImage = AForge.Imaging.Image.Clone(bitmap, PixelFormat.Format24bppRgb);
                tempSource = true;
            }
            else
            {
                sourceImage = bitmap;
            }

            // 1. Приводим к оттенкам серого (результат будет 8bppIndexed)
            IFilter grayFilter = new Grayscale(0.2125, 0.7154, 0.0721);
            Bitmap grayImage = grayFilter.Apply(sourceImage);

            if (tempSource) sourceImage.Dispose();

            // 2. Бинаризация
            Threshold thresholdFilter = new Threshold(BinaryThreshold);
            thresholdFilter.ApplyInPlace(grayImage);

            // 3. Инверсия
            // Для BlobCounter нужны БЕЛЫЕ объекты на ЧЕРНОМ фоне.
            Invert invertFilter = new Invert();
            invertFilter.ApplyInPlace(grayImage);

            // 4. Умная обрезка (Smart Crop) - ищем все куски символа
            Bitmap croppedImage = CropToContent(grayImage);

            // 5. Масштабирование
            Bitmap scaledImage = ResizeWithPadding(croppedImage, InputWidth, InputHeight);

            // 6. Перевод в массив
            double[] input = ConvertBitmapToDoubleArray(scaledImage);

            // Чистим память
            grayImage.Dispose();
            if (croppedImage != grayImage) croppedImage.Dispose();
            scaledImage.Dispose();

            return input;
        }

        public static double[] ConvertBitmapToInput(Bitmap bitmap)
        {
            return ProcessImage(bitmap);
        }


        private static Bitmap CropToContent(Bitmap binImage)
        {
            // BlobCounter работает только с 8bpp или цветными, но мы уже сделали Grayscale (8bpp)
            BlobCounter blobCounter = new BlobCounter
            {
                FilterBlobs = true,
                MinWidth = 2,
                MinHeight = 2,
                ObjectsOrder = ObjectsOrder.None
            };

            blobCounter.ProcessImage(binImage);
            Blob[] blobs = blobCounter.GetObjectsInformation();

            if (blobs.Length == 0)
                return (Bitmap)binImage.Clone();

            int xMin = int.MaxValue, yMin = int.MaxValue;
            int xMax = int.MinValue, yMax = int.MinValue;

            foreach (var blob in blobs)
            {
                if (blob.Rectangle.X < xMin) xMin = blob.Rectangle.X;
                if (blob.Rectangle.Y < yMin) yMin = blob.Rectangle.Y;
                int bRight = blob.Rectangle.X + blob.Rectangle.Width;
                int bBottom = blob.Rectangle.Y + blob.Rectangle.Height;
                if (bRight > xMax) xMax = bRight;
                if (bBottom > yMax) yMax = bBottom;
            }

            Rectangle cropRect = new Rectangle(xMin, yMin, xMax - xMin, yMax - yMin);
            // Защита от выхода за границы
            cropRect.Intersect(new Rectangle(0, 0, binImage.Width, binImage.Height));

            Crop cropFilter = new Crop(cropRect);
            return cropFilter.Apply(binImage);
        }

        private static Bitmap ResizeWithPadding(Bitmap src, int boxWidth, int boxHeight)
        {
            Bitmap result = new Bitmap(boxWidth, boxHeight);
            using (Graphics g = Graphics.FromImage(result))
            {
                g.Clear(Color.Black); // Фон черный (так как мы инвертировали)
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;

                float ratio = Math.Min((float)boxWidth / src.Width, (float)boxHeight / src.Height);
                int newWidth = (int)(src.Width * ratio);
                int newHeight = (int)(src.Height * ratio);

                int posX = (boxWidth - newWidth) / 2;
                int posY = (boxHeight - newHeight) / 2;

                g.DrawImage(src, posX, posY, newWidth, newHeight);
            }
            return result;
        }

        private static double[] ConvertBitmapToDoubleArray(Bitmap bmp)
        {
            double[] result = new double[bmp.Width * bmp.Height];
            for (int y = 0; y < bmp.Height; y++)
            {
                for (int x = 0; x < bmp.Width; x++)
                {
                    Color pixel = bmp.GetPixel(x, y);
                    result[y * bmp.Width + x] = pixel.R / 255.0;
                }
            }
            return result;
        }

        public static Bitmap GenerateVariatedImage(Bitmap original, Random rnd)
        {
            // Здесь тоже может быть ошибка формата, если RotateBicubic получит не тот формат.
            // Поэтому тоже защитим конвертацией
            Bitmap source = original;
            bool temp = false;
            if (source.PixelFormat != PixelFormat.Format24bppRgb &&
                source.PixelFormat != PixelFormat.Format8bppIndexed &&
                source.PixelFormat != PixelFormat.Format32bppArgb)
            {
                source = AForge.Imaging.Image.Clone(original, PixelFormat.Format24bppRgb);
                temp = true;
            }

            RotateBicubic rotateFilter = new RotateBicubic(rnd.Next(-10, 10), true);
            rotateFilter.FillColor = Color.White;

            Bitmap result = rotateFilter.Apply(source);

            if (temp) source.Dispose();
            return result;
        }

        public static Bitmap CreateBitmapFromInput(double[] input, int width, int height)
        {
            Bitmap bmp = new Bitmap(width, height);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int val = (int)(input[y * width + x] * 255);
                    val = Math.Max(0, Math.Min(255, val));
                    bmp.SetPixel(x, y, Color.FromArgb(val, val, val));
                }
            }
            return bmp;
        }
    }
}