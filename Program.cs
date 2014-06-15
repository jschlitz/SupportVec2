using Microsoft.SolverFoundation.Solvers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.SolverFoundation.Common;

namespace SV2
{
  class Program
  {
    static Random R = new Random();
    const int TRIALS = 100;
    const int EXPERIMENTS = 100;
    const int THRESH = 100000000;

    static void Main(string[] args)
    {
      var wTarget = (new[] { 0.6, 0.8 });
      var C = (Rational) 10.0;
      var w0Target = 0.1;
      var rightness = 0.0;
      var successful = 0.0;

      for (int experiment = 0; experiment < EXPERIMENTS; experiment++)
      {
        try
        {
          var classified = Generate(TRIALS, w0Target, wTarget);
          var testData = Generate(TRIALS * 10, w0Target, wTarget);

          var solver = new InteriorPointSolver();
          Func<double[], double[], double> kernel = Dot;
          int goal;
          solver.AddRow("dual", out goal);
          solver.AddGoal(goal, 0, false);//false to maximize

          //make alphas
          var alphas = classified.Select((_, i) =>
          {
            int tmp;
            solver.AddVariable("alpha" + i, out tmp);
            solver.SetBounds(tmp, Rational.Zero, C); 
            return tmp;
          }).ToArray();

          int sumConstraint;
          solver.AddRow("sumConstraint", out sumConstraint);
          //solver.SetBounds(sumConstraint, Rational.Zero, Rational.Zero);
          //TODO: maybe I'm running into numeric issues?
          solver.SetBounds(sumConstraint, Rational.Get(-1, THRESH), Rational.Get(1, THRESH));
          for (int i = 0; i < classified.Length; i++)
          {
            //sum_n (alpha_n*y_n) == 0
            solver.SetCoefficient(sumConstraint, alphas[i], classified[i].y);

            solver.SetCoefficient(goal, alphas[i], Rational.One); //the sum_n (alpha_n) part of the lagrangian

            ////quadratic terms. sometimes not convex, and I don't know why
            for (int j = 0; j <= i; j++)
            {
              // coef = y_i * y_j * Kernel(x_i, x_j). Note that the diagonal is half: the other terms appear twice and are thus doubled
              var coef = (i == j ? -0.5 : -1.0) * classified[i].y * classified[j].y * kernel(classified[i].x, classified[j].x);
              solver.SetCoefficient(goal, coef, alphas[i], alphas[j]);
            }
            //This gives better results, but it isn't actually right. :-(
            //for (int j = 0; j < classified.Length; j++)
            //{
            //  // coef = y_i * y_j * Kernel(x_i, x_j). Note that the diagonal is half: the other terms appear twice and are thus doubled
            //  var coef = -0.5  * classified[i].y * classified[j].y * kernel(classified[i].x, classified[j].x);
            //  solver.SetCoefficient(goal, coef, alphas[i], alphas[j]);
            //}
          }

          //now solve
          solver.Solve(new InteriorPointSolverParams());
          var alphaVals = Enumerable.Range(0, classified.Length)
            .Select(i => solver.GetValue(alphas[i])) //using alphas[i] instead of 1..N here
            .ToArray();

          //Console.WriteLine("goal={0}", solver.GetValue(0));

          //get the SVs
          var maxAlpha = alphaVals.Max();
          var threshold = maxAlpha / THRESH;
          var sVecs = alphaVals.Where(a => a > threshold)
            //.Where(a => a + threshold < C) //This may be wrong... yep
            .Select((a, i) => new SVInfo { y = classified[i].y, x = classified[i].x, alpha = a })
            .ToArray();

          //w0, aka b
          var w0s = sVecs.Where(sv => sv.alpha + threshold < C) //this must not be right?
          .Select(sv1 =>
            //note that the inner loop gets alphas == C. Not sure if that's right.
              sv1.y - sVecs.Select(sv2 => sv2.alpha.ToDouble() * sv2.y * kernel(sv1.x, sv2.x)).Sum()
            ).ToArray();

          var w0 = w0s.OrderBy(lf => lf).ToArray()[w0s.Length / 2]; //too widely dispersed... median here
          //so what does w look like here...?
          List<double> wTmp = (new List<double> { w0 });
          wTmp.AddRange(
            sVecs.Select(sv =>
          {
            var tmp = sv.alpha.ToDouble() * sv.y;
            return sv.x.Select(x => x * tmp);
          }).Aggregate((acc, xs) => acc.Zip(xs, (a, b) => a + b)));
          var wOut = Norm(wTmp.ToArray());





          var classifiedTests = Classify(sVecs, testData, kernel, w0);//They filter out things athat are = C
          //var classifiedTests = Classify(sVecs.Where(sv => sv.alpha + threshold < C).ToArray(), testData, kernel, w0);          

          var right = 100.0 * ((double)classifiedTests.Count(b => b)) / ((double)classifiedTests.Length);
          Console.Write(".");
          //Console.WriteLine("Correctly classified {0}%", right);
          rightness += right;
          successful += 1.0;
        }
        catch
        { Console.WriteLine("X"); }
      }

      Console.WriteLine("Total rightness {0}%", rightness/successful);
      Console.ReadKey(true);
    }

    private static bool[] Classify(SVInfo[] sVecs, Datum[] testData, Func<double[], double[], double> kernel, double w0)
    {
      var foo = testData
        .Select(d => {
          var tmp = sVecs.Select(sv => sv.alpha.ToDouble() * sv.y * kernel(sv.x, d.x)).Sum() + w0;
          return tmp * d.y > 0;//numeric instability mignt make tmp not qute be -1/+1.
        })
        .ToArray();
      return foo;
    }

    private struct SVInfo
    {
      public double y;
      public double[] x;
      public Rational alpha;
    }

    private struct Datum
    {
      public double y;
      public double[] x;
    }

    private static double[] Norm(double[] wPla)
    {
      var mag = Math.Sqrt(wPla.Sum(x => x * x));
      return wPla.Select(x => x / mag).ToArray();
    }

    private static Datum[] Generate(int count, double w0, double[] w)
    {
      return Enumerable.Range(1, count)
        .Select(_ => GetX())
        .Select(ex => new Datum{x=ex, y=GetY(ex, w0, w)})
        .ToArray();
    }

    private static double GetY(double[] x, double w0, double[] w)
    {
      return (Dot(x, w) + w0 < 0) ? -1.0 : 1.0;
    }

    private static double Dot(double[] v1, double[] v2)
    {
      return v1.Zip(v2, (v1n, v2n) => v1n * v2n).Sum();
    }

    private static double[] GetX()
    {
      return new[] { GetR(), GetR() };
    }

    private static double GetR()
    {
      return R.NextDouble() * 2 - 1;
    }


  }
}
