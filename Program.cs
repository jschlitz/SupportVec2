using Microsoft.SolverFoundation.Solvers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.SolverFoundation.Common;
using System.IO;

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
      var wTarget = (new[] { Rational.Get(6, 10), Rational.Get(8, 10), Rational.Get(-5,10) });
      var C = Rational.Get(10,1);
      var w0Target = Rational.Get(1,10);
      var rightness = 0.0;
      var successful = 0.0;
      Datum[] classified={};
      var dump = new Rational[TRIALS,TRIALS];

      for (int experiment = 0; experiment < EXPERIMENTS; experiment++)
      {
        try
        {
          classified = Generate(TRIALS, w0Target, wTarget);
          var testData = Generate(TRIALS * 10, w0Target, wTarget);

          var solver = new InteriorPointSolver();
          Func<Rational[], Rational[], Rational> kernel = Dot;
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

            solver.SetCoefficient(goal, alphas[i], 1); //the sum_n (alpha_n) part of the lagrangian

            ////quadratic terms. sometimes not convex, and I don't know why
            for (int j = 0; j <= i; j++)
            {
              // coef = y_i * y_j * Kernel(x_i, x_j). Note that the diagonal is half: the other terms appear twice and are thus doubled
              var coef = (i == j ? -0.5 : -1.0) * classified[i].y * classified[j].y * kernel(classified[i].x, classified[j].x);
              solver.SetCoefficient(goal, coef, alphas[i], alphas[j]);
              dump[i, j] = coef;
              dump[j, i] = coef;
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
              sv1.y - sVecs.Select(sv2 => sv2.alpha * sv2.y * kernel(sv1.x, sv2.x)).Aggregate((acc, r) => acc + r)
            ).ToArray();

          var w0 = w0s.OrderBy(lf => lf.ToDouble()).ToArray()[w0s.Length / 2]; //too widely dispersed... median here
          //so what does w look like here...?
          List<Rational> wTmp = (new List<Rational> { w0 });
          wTmp.AddRange(
            sVecs.Select(sv =>
          {
            var tmp = sv.alpha * sv.y;
            return sv.x.Select(x => x * tmp);
          }).Aggregate((acc, xs) => acc.Zip(xs, (a, b) => a + b)));
          var wOut = Norm(wTmp.ToArray());

          var classifiedTests = Classify(sVecs, testData, kernel, w0);//They filter out things athat are = C
          //var classifiedTests = Classify(sVecs.Where(sv => sv.alpha + threshold < C).ToArray(), testData, kernel, w0);          

          var right = 100.0 * (classifiedTests.Count(b => b)) / (classifiedTests.Length);
          Console.Write(".");
          //Console.WriteLine("Correctly classified {0}%", right);
          rightness += right;
          successful += 1.0;
        }
        catch (Exception ex)
        {
          var sb = new StringBuilder();
          for (int i = 0; i < TRIALS; i++)
          {
            for (int j = 0; j < TRIALS-1; j++)
            {
              sb.Append(dump[i, j]);
              sb.Append(",");
            }
            sb.AppendLine(dump[i, TRIALS - 1].ToString());
          }

          sb.AppendLine();
          foreach (var item in classified)
            sb.AppendLine(item.ToString());

          var p = Path.Combine(Environment.GetFolderPath( Environment.SpecialFolder.CommonApplicationData), 
            "SV2",
            DateTime.Now.ToString("yyyyMMdd-HHmmss-") + experiment + ".csv");
          Directory.CreateDirectory(Path.GetDirectoryName(p));
          Console.WriteLine(p);
          using (var sr = new StreamWriter(p))
            sr.Write(sb.ToString());
        }
      }

      Console.WriteLine("Total rightness {0}%", rightness/successful);
      Console.ReadKey(true);
    }

    private static bool[] Classify(SVInfo[] sVecs, Datum[] testData, Func<Rational[], Rational[], Rational> kernel, Rational w0)
    {
      var foo = testData
        .Select(d => {
          var tmp = sVecs.Select(sv => sv.alpha * sv.y * kernel(sv.x, d.x)).Aggregate((acc, r) => acc + r) + w0;
          return tmp * d.y > 0;//numeric instability mignt make tmp not qute be -1/+1.
        })
        .ToArray();
      return foo;
    }

    private struct SVInfo
    {
      public Rational y;
      public Rational[] x;
      public Rational alpha;
      public override string ToString()
      {
        return string.Join(",", x) + "," + y.ToString() + "," + alpha.ToString();
      }
    }

    private struct Datum
    {
      public Rational y;
      public Rational[] x;
      public override string ToString()
      {
        return string.Join(",", x) + "," + y.ToString();
      }
    }

    private static Rational[] Norm(Rational[] wPla)
    {
      var mag = (Rational) Math.Sqrt(wPla.Aggregate((acc, r)=>acc+r).ToDouble());
      return wPla.Select(x => x / mag).ToArray();
    }

    private static Datum[] Generate(int count, Rational w0, Rational[] w)
    {
      return Enumerable.Range(1, count)
        .Select(_ => GetX())
        .Select(ex => new Datum{x=ex, y=GetY(ex, w0, w)})
        .ToArray();
    }

    private static Rational GetY(Rational[] x, Rational w0, Rational[] w)
    {
      return (Dot(x, w) + w0 < 0) ? -1.0 : 1.0;
    }

    private static Rational Dot(Rational[] v1, Rational[] v2)
    {
      return v1.Zip(v2, (v1n, v2n) => v1n * v2n).Aggregate((acc, r)=>acc+r);
    }

    private static Rational[] GetX()
    {
      return new[] { GetR(), GetR() };
    }

    private static Rational GetR()
    {
      return Rational.Get(R.Next(-1000,1000),1000);
    }
  }
}
